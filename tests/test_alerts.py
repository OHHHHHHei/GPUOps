from gpuops import alerts
from gpuops.config import ServerTarget


def test_alert_poll_reports_only_confirmed_stop_and_recovery(tmp_path, monkeypatch):
    servers = [
        ServerTarget("xmu06", "Server 6", "six"),
        ServerTarget("xmu11", "Server 11", "eleven"),
    ]
    healthy = {"status": "running", "ready": True, "reason": "ready"}
    stopped = {
        "status": "stopped",
        "ready": False,
        "reason": "worker process is stopped",
        "heartbeat": {
            "heartbeat_at": "2026-11-06T07:53:58Z",
            "last_tick_error": "No space left on device",
        },
    }
    samples = iter([healthy, None, stopped, stopped, None, {"ready": False}, healthy, healthy])
    messages = []

    async def collect(_transport, targets, action):
        assert [target.server_id for target in targets] == ["xmu06", "xmu11"]
        assert action == "status"
        sample = next(samples)
        other = (
            {"online": False, "error": "SSH timed out"}
            if sample is None
            else {"online": True, "result": sample}
        )
        return [
            {"server_id": "xmu06", "online": True, "result": healthy},
            {"server_id": "xmu11", **other},
        ]

    monkeypatch.setattr(alerts, "select_targets", lambda: servers)
    monkeypatch.setattr(alerts, "fleet_worker_action", collect)
    monkeypatch.setattr(alerts, "_notify", lambda title, text: messages.append((title, text)))

    path = tmp_path / "alerts.json"
    states = [alerts._check(path)["servers"][1]["status"] for _ in range(8)]
    assert states == [
        "ready",
        "unknown",
        "stopped",
        "stopped",
        "unknown",
        "not_ready",
        "ready",
        "ready",
    ]
    assert len(messages) == 2
    assert messages[0][0] == messages[1][0] == "GPUOps: Server 11"
    assert "No space left on device" in messages[0][1]
    assert "Last heartbeat:" in messages[0][1]
    assert "recovered" in messages[1][1]
