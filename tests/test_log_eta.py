import asyncio
import json
import os
import sys
import time
from datetime import datetime

import pytest

from gpuops.config import ServerTarget
from gpuops.monitor.service import ServerResult
from gpuops.ssh import SSHResult
from gpuops.web import _log_sample
from gpuops.web.log_eta import LogEtaCollector
from gpuops.web.progress import estimate
from gpuops.web.provider import LiveProvider


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


NOW = stamp("2026-09-20T23:18:19+08:00")
EPOCH_LOG = """09/20/26 23:11:59 - Epoch 1, Average Loss: 0.51, train time: 3.1min
09/20/26 23:14:51 - Epoch 2, Average Loss: 0.35, train time: 2.8min
09/20/26 23:14:52 - Epoch 2 Validation | AUC: 0.65
09/20/26 23:17:42 - Epoch 3, Average Loss: 0.31, train time: 2.8min
"""


def sample(content=EPOCH_LOG, argv=None):
    process = {
        "pid": 123,
        "start_time_ticks": 456,
        "started_at": NOW - 1000,
        "argv": argv if argv is not None else ["python", "train.py", "--num_epochs", "15"],
        "paths": ["/logs/train.log"],
    }
    return {
        "sampled_at": NOW,
        "utc_offset_seconds": 28800,
        "processes": [process],
        "logs": {"/logs/train.log": {"content": content, "modified_at": NOW - 30}},
    }


def predict(value, now=NOW):
    return estimate(value, value["processes"][0], "/logs/train.log", now=now)


def test_epoch_eta_replays_real_timing_and_includes_inter_epoch_overhead():
    result = predict(sample())
    assert result["end_at"] == "2026-09-20T15:52:00+00:00"
    assert result["scope"] == "training"
    assert "3/15 epoch" in result["basis"]
    assert result["source"] == "/logs/train.log"


@pytest.mark.parametrize("argv", [["--epochs=15"], []])
def test_total_epochs_can_come_from_argv_or_logged_configuration(argv):
    assert predict(sample("Args: {'num_epochs': 15}\n" + EPOCH_LOG, argv))["end_at"]


def test_epoch_summary_with_reported_seconds_needs_no_timestamp():
    value = sample(
        "trial epoch=1/3 loss=3.9 seconds=946.1\ntrial epoch=2/3 loss=3.8 seconds=866.8\n", []
    )
    result = predict(value)
    assert result["scope"] == "training"
    assert stamp(result["end_at"]) == NOW - 30 + (946.1 + 866.8) / 2
    assert predict(value, NOW + 60)["end_at"] == result["end_at"]


def test_per_batch_elapsed_seconds_do_not_mean_an_epoch_has_completed():
    content = "epoch=1/3 microbatch=100/1000 elapsed_sec=100\n"
    assert predict(sample(content, []))["end_at"] is None


def test_followup_output_does_not_redate_an_untimestamped_epoch_summary():
    content = "epoch=1/3 seconds=100\nStarting validation...\n"
    assert predict(sample(content, []))["end_at"] is None


def test_validation_duration_is_not_a_whole_training_duration():
    assert predict(sample("validation epoch=2/3 seconds=1.5\n", []))["end_at"] is None


def test_early_stopping_and_training_completion_invalidate_old_progress():
    for marker in ("Training complete!", "Early stopping"):
        assert predict(sample(EPOCH_LOG + marker))["end_at"] is None


@pytest.mark.parametrize(
    "content",
    [
        "loss=0.3\nprogress 55%\nfeatures 120/256",
        "Epoch 1/15\nEpoch 2/15\n",  # epoch headers are not completion evidence
        "09/20/26 23:17:42 - Epoch 3, Average Loss: 0.31\n",
        EPOCH_LOG + "09/20/26 23:18:00 - Epoch 1 done\n",  # reset/new stage
    ],
)
def test_ambiguous_progress_never_becomes_an_eta(content):
    assert predict(sample(content))["end_at"] is None


def test_stale_completed_and_previous_run_logs_are_not_predictions():
    assert predict(sample(), NOW + 1800)["end_at"] is None
    complete = EPOCH_LOG + "09/20/26 23:18:00 - Epoch 15 done\n"
    assert predict(sample(complete))["end_at"] is None
    previous = sample()
    previous["processes"][0]["started_at"] = NOW
    assert predict(previous)["end_at"] is None


@pytest.mark.parametrize("name,scope", [("global_step", "training"), ("step", "phase")])
def test_timestamped_steps_estimate_speed_with_explicit_scope(name, scope):
    content = f"2026-09-20T15:16:00Z {name} 100/1000\n2026-09-20T15:17:00Z {name} 200/1000"
    result = predict(sample(content, []))
    assert result["end_at"] == "2026-09-20T15:25:00+00:00"
    assert result["scope"] == scope


def test_step_counter_reset_does_not_mix_speeds_across_epochs():
    content = "2026-09-20T15:16:00Z step 100/1000\n2026-09-20T15:17:00Z step 200/1000\n"
    content += "2026-09-20T15:18:00Z step 10/1000"
    assert predict(sample(content, []))["end_at"] is None


@pytest.mark.parametrize("speed", ["2.00it/s", "0.50s/it"])
def test_tqdm_carriage_returns_ansi_and_polling_do_not_slide_the_end_time(speed):
    content = f"\r 10%|xx| 100/1000 [00:50<07:30, {speed}]"
    content += f"\r\x1b[32m20%|xxxx| 200/1000 [01:40<06:40, {speed}]\x1b[0m"
    value = sample(content, [])
    first = predict(value)
    second = predict(value, NOW + 60)
    assert stamp(first["end_at"]) == NOW - 30 + 400
    assert first["end_at"] == second["end_at"]
    assert first["scope"] == "phase"
    assert predict(value, NOW + 500)["end_at"] is None


def test_untimestamped_bar_followed_by_other_output_has_no_reliable_anchor():
    content = "200/1000 [01:40<06:40, 2.00it/s]\nSaving checkpoint..."
    assert predict(sample(content))["end_at"] is None


def test_sampler_deduplicates_ddp_logs_checks_identity_and_never_reads_pipes(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "uptime").write_text("1000 0")
    directory = tmp_path / "logs"
    directory.mkdir()
    log = directory / "train.log"
    log.write_text(EPOCH_LOG)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for pid in (101, 102):
        root = proc / str(pid)
        (root / "fd").mkdir(parents=True)
        fields = ["S", "1", *(["0"] * 17), "456"]
        (root / "stat").write_text(f"{pid} (python) " + " ".join(fields))
        (root / "cmdline").write_bytes(f"python\0train.py\0--log_dir={directory}\0".encode())
        (root / "fd/1").symlink_to(fifo)
    monkeypatch.setattr(_log_sample, "PROC", proc)
    read_paths = []
    read = _log_sample.read_log

    def record_read(path):
        read_paths.append(path)
        return read(path)

    monkeypatch.setattr(_log_sample, "read_log", record_read)
    result = _log_sample.collect(
        [
            {"pid": 101, "start_time_ticks": 456},
            {"pid": 102, "start_time_ticks": 456},
            {"pid": 101, "start_time_ticks": 999},
        ]
    )
    assert read_paths.count(str(log)) == 1
    assert result["logs"][str(log)]["content"] == EPOCH_LOG
    assert result["logs"][str(fifo)]["error"] == "NotRegularFile"
    assert result["processes"][-1]["error"] == "ProcessChanged"


def test_large_log_sampling_keeps_header_and_tail_with_bounded_reads(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(b"Args: {'epochs': 15}\n" + b"padding\n" * 40000 + EPOCH_LOG.encode())
    result = _log_sample.read_log(log)
    assert result["truncated"]
    assert len(result["content"].encode()) <= _log_sample.HEAD_BYTES + _log_sample.TAIL_BYTES + 1
    assert result["content"].startswith("Args:") and result["content"].endswith(EPOCH_LOG)


@pytest.fixture
def managed_process_tree(tmp_path, monkeypatch):
    from gpuops.worker.supervisor import ProcessIdentity, SupervisorReceipt

    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "uptime").write_text("1000 0")
    boot_directory = proc / "sys/kernel/random"
    boot_directory.mkdir(parents=True)
    (boot_directory / "boot_id").write_text("test-boot\n")
    run = tmp_path / "run"
    run.mkdir()
    log = run / "console.log"
    log.write_text(EPOCH_LOG)
    intent = {"run_id": "test-run", "launch_token": "test-token", "console_path": str(log)}
    (run / "intent.json").write_text(json.dumps(intent))
    receipt = SupervisorReceipt(
        run_id="test-run",
        launch_token="test-token",
        status="running",
        accepted_at="2026-09-21T00:00:00+00:00",
        supervisor=ProcessIdentity(100, 100, 450, "test-boot"),
        child=ProcessIdentity(101, 101, 456, "test-boot"),
    ).as_json_dict()
    (run / "receipt.json").write_text(json.dumps(receipt))
    for pid, parent, ticks, argv in (
        (
            100,
            1,
            450,
            [
                "python",
                "/pkg/gpuops/worker/supervisor.py",
                "_supervise",
                "--intent",
                str(run / "intent.json"),
            ],
        ),
        (101, 100, 456, ["python", "-m", "torch.distributed.run", "train.py"]),
        (102, 101, 460, ["python", "train.py"]),
    ):
        root = proc / str(pid)
        (root / "fd").mkdir(parents=True)
        fields = ["S", str(parent), *(["0"] * 17), str(ticks)]
        (root / "stat").write_text(f"{pid} (python) " + " ".join(fields))
        (root / "cmdline").write_bytes("\0".join(argv).encode())
        (root / "fd/1").symlink_to("pipe:[1234]")
    monkeypatch.setattr(_log_sample, "PROC", proc)
    return run, receipt


def test_managed_pipe_output_is_resolved_for_child_and_ddp_descendants(managed_process_tree):
    run, _ = managed_process_tree
    result = _log_sample.collect(
        [
            {"pid": 101, "start_time_ticks": 456},
            {"pid": 102, "start_time_ticks": 460},
        ]
    )
    assert len(result["logs"]) == 1
    for row in result["processes"]:
        assert row["paths"] == [str(run / "console.log")]
        assert row["run_id"] == "test-run"


@pytest.mark.parametrize("change", ["child", "supervisor", "launch_token", "run_id", "boot_id"])
def test_stale_receipt_never_associates_another_runs_log(managed_process_tree, change):
    run, receipt = managed_process_tree
    if change in {"child", "supervisor"}:
        receipt[change]["start_ticks"] += 1
    elif change == "boot_id":
        receipt["supervisor"]["boot_id"] = receipt["child"]["boot_id"] = "previous-boot"
    else:
        receipt[change] = "different-run"
    (run / "receipt.json").write_text(json.dumps(receipt))
    result = _log_sample.collect([{"pid": 102, "start_time_ticks": 460}])
    assert not result["processes"][0]["paths"]
    assert not result["logs"]


def test_unreadable_launch_metadata_keeps_normal_log_discovery(managed_process_tree):
    run, _ = managed_process_tree
    (run / "receipt.json").write_text("partial")
    root = _log_sample.PROC / "102"
    (root / "fd/2").symlink_to(run / "console.log")
    result = _log_sample.collect([{"pid": 102, "start_time_ticks": 460}])
    assert result["processes"][0]["paths"] == [str(run / "console.log")]
    assert "run_id" not in result["processes"][0]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux /proc")
def test_real_supervisor_pipe_is_sampled_and_yields_structured_eta(tmp_path):
    from gpuops.worker.supervisor import DurableJobSupervisor

    # A CPU-only child exercises the real launch receipt, pipe, and console writer.
    script = """
import json, time
from pathlib import Path
started = time.monotonic()
for current in (0, 1):
    print('GPUOPS_PROGRESS ' + json.dumps(dict(phase='training', current=current,
        total=100000, elapsed_seconds=time.monotonic()-started, timestamp=time.time())),
        flush=True)
    time.sleep(0.02)
while not Path('stop').exists():
    time.sleep(0.01)
"""
    supervisor = DurableJobSupervisor(tmp_path / "state")
    receipt = supervisor.launch("progress-test", [sys.executable, "-u", "-c", script], cwd=tmp_path)
    try:
        request = {"pid": receipt.child.pid, "start_time_ticks": receipt.child.start_ticks}
        deadline = time.monotonic() + 5
        result = None
        while time.monotonic() < deadline:
            value = _log_sample.collect([request])
            process = value["processes"][0]
            if process["paths"]:
                result = estimate(value, process, process["paths"][0], now=time.time())
                if result.get("end_at"):
                    break
            time.sleep(0.01)
        assert process["run_id"] == "progress-test"
        assert result["format"] == "gpuops_progress"
        assert result["end_at"]
        assert result["progress"] == {"phase": "training", "current": 1, "total": 100000}
    finally:
        (tmp_path / "stop").touch()
        assert supervisor.wait("progress-test", timeout_seconds=5).exit_code == 0


def test_web_uses_cached_log_samples_and_a_new_process_invalidates_them(monkeypatch):
    monkeypatch.setattr("gpuops.web.log_eta.time.time", lambda: NOW)

    class Ssh:
        calls = 0

        async def run_python(self, host, script, *, timeout_seconds):
            self.calls += 1
            return SSHResult(host, 0, json.dumps(sample()), "", 0.1)

    ssh = Ssh()
    collector = LogEtaCollector(ssh=ssh)
    raw = {"index": 0, "processes": [{"pid": 123, "start_time_ticks": 456}]}
    result = ServerResult(ServerTarget("test", "Test", "test"), True, data={"gpus": [raw]})

    class Monitor:
        async def collect_all(self):
            return [result]

    async def check():
        provider = LiveProvider(Monitor(), log_eta=collector, expected_gpu_count=1)
        first = await provider.get_process_etas()
        await provider.get_process_etas()
        assert ssh.calls == 1
        eta = first["servers"][0]["processes"][0]["eta"]
        assert eta["end_at"] == "2026-09-20T15:52:00+00:00"
        raw["processes"][0]["start_time_ticks"] = 789
        await provider.get_process_etas()
        assert ssh.calls == 2

    asyncio.run(check())


def test_structured_unknown_takes_precedence_over_legacy_eta_in_another_log(monkeypatch):
    monkeypatch.setattr("gpuops.web.log_eta.time.time", lambda: NOW)
    value = sample()
    value["processes"][0]["paths"].append("/logs/progress.log")
    value["logs"]["/logs/progress.log"] = {"content": "GPUOPS_PROGRESS {invalid"}

    class Ssh:
        async def run_python(self, host, script, *, timeout_seconds):
            return SSHResult(host, 0, json.dumps(value), "", 0.1)

    result = ServerResult(
        ServerTarget("test", "Test", "test"),
        True,
        data={"gpus": [{"processes": [{"pid": 123, "start_time_ticks": 456}]}]},
    )
    estimates = asyncio.run(LogEtaCollector(ssh=Ssh()).estimates([result]))
    assert estimates["test"][123]["format"] == "gpuops_progress"
    assert estimates["test"][123]["end_at"] is None


def test_one_hosts_log_failure_does_not_hide_its_gpu_snapshot(monkeypatch):
    monkeypatch.setattr("gpuops.web.log_eta.time.time", lambda: NOW)

    class Ssh:
        async def run_python(self, host, script, *, timeout_seconds):
            if host == "failed":
                raise TimeoutError("offline")
            return SSHResult(host, 0, json.dumps(sample()), "", 0.1)

    class Monitor:
        async def collect_all(self):
            return [
                ServerResult(
                    ServerTarget(host, host, host),
                    True,
                    data={
                        "gpus": [
                            {
                                "index": 0,
                                "processes": [{"pid": 123, "start_time_ticks": 456}],
                                "utilization.gpu": 100,
                                "memory.used": 10000,
                            }
                        ]
                    },
                )
                for host in ("good", "failed")
            ]

    provider = LiveProvider(Monitor(), log_eta=LogEtaCollector(ssh=Ssh()), expected_gpu_count=1)
    output = asyncio.run(provider.get_snapshot())
    estimates = asyncio.run(provider.get_process_etas())
    assert estimates["servers"][0]["processes"][0]["eta"]["end_at"]
    failed_gpu = output["servers"][1]["gpus"][0]
    assert failed_gpu["status"] == "busy"
    assert failed_gpu["processes"][0]["eta"]["end_at"] is None


def test_slow_log_sampling_does_not_delay_gpu_snapshots_or_health():
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        class SlowLogs:
            async def estimates(self, results):
                entered.set()
                await release.wait()
                return {}

        class Monitor:
            async def collect_all(self):
                return [ServerResult(ServerTarget("test", "Test", "test"), True, data={"gpus": []})]

        provider = LiveProvider(Monitor(), log_eta=SlowLogs())
        sampling = asyncio.create_task(provider.get_process_etas())
        await entered.wait()
        try:
            snapshot = await asyncio.wait_for(provider.get_snapshot(), 0.2)
            health = await asyncio.wait_for(provider.health(), 0.2)
            assert snapshot["servers"][0]["id"] == "test"
            assert health["servers"][0]["id"] == "test"
            assert not sampling.done()
        finally:
            release.set()
            await sampling

    asyncio.run(check())
