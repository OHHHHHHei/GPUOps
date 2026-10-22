import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "gpuops-deploy/v1"
TOTAL_BUDGET = 1024 * 1024 * 1024
STATE_BUDGET = 768 * 1024 * 1024
SHARE_BUDGET = 128 * 1024 * 1024
CACHE_BUDGET = 128 * 1024 * 1024
INSTALL_BUDGET = 256 * 1024 * 1024
HEARTBEAT_MAX_AGE_SECONDS = 75.0
WORKER_LIFECYCLE_TIMEOUT_SECONDS = 75.0
ROLLBACK_COMPATIBILITY_TIMEOUT_SECONDS = 30.0
RELEASE_ID_PATTERN = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{64})$")
ARTIFACT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*\.(?:whl|tar\.gz)$")
DEPLOYMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SERVER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
ACTION = globals().get("ACTION")
RELEASE_ID = sys.argv[1] if ACTION == "deploy" and len(sys.argv) > 1 else None
DEPLOYMENT_ID = sys.argv[2] if ACTION == "deploy" and len(sys.argv) > 2 else None
REQUESTED_SERVER_ID = sys.argv[3] if ACTION == "deploy" and len(sys.argv) > 3 else None
SSH_HOST = sys.argv[4] if ACTION == "deploy" and len(sys.argv) > 4 else None
home = Path.home()
cache = home / ".cache" / "gpuops"
share = home / ".local" / "share" / "gpuops"
state = home / ".local" / "state" / "gpuops"
releases = share / "releases"
current = share / "current"
local_bin = home / ".local" / "bin"
executable = local_bin / "gpuops"
expected_executable_target = "../share/gpuops/current/venv/bin/gpuops"
lock_path = share / "deploy.lock"
host_identity_path = share / "host.json"
worker_pid_path = state / "worker.pid.json"
worker_heartbeat_path = state / "worker-health.json"
os.umask(0o077)


def atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_symlink(path, target):
    temporary = path.with_name("." + path.name + "." + str(os.getpid()))
    temporary.unlink(missing_ok=True)
    os.symlink(target, temporary)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def directory_size(path):
    if not path.exists():
        return 0
    total = 0
    for root, directories, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories + filenames:
            candidate = root_path / name
            with suppress(FileNotFoundError):
                total += candidate.lstat().st_size
    return total


def usage():
    state_bytes = directory_size(state)
    share_bytes = directory_size(share)
    cache_bytes = directory_size(cache)
    return state_bytes, share_bytes, cache_bytes


def ensure_owned_budget():
    state_bytes, share_bytes, cache_bytes = usage()
    install_bytes = share_bytes + cache_bytes
    if state_bytes > STATE_BUDGET:
        raise RuntimeError("GPUOps state directory exceeds its 768 MiB budget")
    if share_bytes > SHARE_BUDGET:
        raise RuntimeError("GPUOps share directory exceeds its 128 MiB budget")
    if cache_bytes > CACHE_BUDGET:
        raise RuntimeError("GPUOps cache directory exceeds its 128 MiB budget")
    if install_bytes > INSTALL_BUDGET:
        raise RuntimeError("GPUOps install/cache directories exceed their 256 MiB budget")
    if state_bytes + install_bytes > TOTAL_BUDGET:
        raise RuntimeError("GPUOps owned directories exceed the global 1 GiB budget")


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_timestamp(value):
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def process_identity(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        boot_id = (Path("/proc/sys/kernel/random/boot_id")).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 1 :].split() if closing >= 0 else []
    if len(fields) <= 19:
        return None
    try:
        return {
            "pid": pid,
            "start_time_ticks": int(fields[19]),
            "boot_id": boot_id,
        }
    except ValueError:
        return None


def stored_worker_identity():
    value = read_json(worker_pid_path)
    if value is None:
        return None
    try:
        return {
            "pid": int(value["pid"]),
            "start_time_ticks": int(value["start_time_ticks"]),
            "boot_id": str(value["boot_id"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def worker_snapshot(expected_server_id=None):
    stored = stored_worker_identity()
    heartbeat = read_json(worker_heartbeat_path)
    matched = stored is not None and process_identity(stored["pid"]) == stored
    ready = False
    reason = "worker process identity is unavailable"
    if matched and heartbeat is not None:
        try:
            heartbeat_at = parse_timestamp(heartbeat["heartbeat_at"])
            now = datetime.now(timezone.utc)
            if heartbeat_at.tzinfo is None:
                raise ValueError("heartbeat is not timezone-aware")
            age = (now - heartbeat_at).total_seconds()
            ready = (
                heartbeat.get("status") == "running"
                and int(heartbeat.get("pid")) == stored["pid"]
                and (expected_server_id is None or heartbeat.get("server_id") == expected_server_id)
                and -5.0 <= age <= HEARTBEAT_MAX_AGE_SECONDS
            )
            reason = "ready" if ready else "worker heartbeat is not ready"
        except (KeyError, TypeError, ValueError):
            reason = "worker heartbeat is invalid"
    return {
        "process_running": matched,
        "ready": ready,
        "reason": reason,
        "pid": stored["pid"] if matched else None,
        "server_id": heartbeat.get("server_id") if heartbeat else None,
    }


def release_for_pid(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    prefix = str(releases.resolve()) + "/"
    for raw in arguments:
        argument = raw.decode("utf-8", errors="ignore")
        position = argument.find(prefix)
        if position < 0:
            continue
        release_id = argument[position + len(prefix) :].split("/", 1)[0]
        if RELEASE_ID_PATTERN.fullmatch(release_id):
            return release_id
    return None


def live_release_ids(exclude_pid=None):
    live = set()
    for process in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process.name)
        except ValueError:
            continue
        if pid == exclude_pid:
            continue
        release_id = release_for_pid(pid)
        if release_id is not None:
            live.add(release_id)
    return live


def installed_server_id():
    value = read_json(host_identity_path)
    if value is None:
        if host_identity_path.exists():
            raise RuntimeError("installed host identity is invalid")
        return None
    server_id = value.get("server_id")
    if (
        not isinstance(server_id, str)
        or not SERVER_ID_PATTERN.fullmatch(server_id)
        or server_id == "auto"
    ):
        raise RuntimeError("installed host identity has an invalid server id")
    return server_id


def deployment_server_id(requested=None):
    if requested == "":
        requested = None
    if requested is not None and (
        not isinstance(requested, str)
        or not SERVER_ID_PATTERN.fullmatch(requested)
        or requested == "auto"
    ):
        raise RuntimeError("requested server id is invalid")
    installed = installed_server_id()
    if installed is not None:
        if requested is not None and requested != installed:
            raise RuntimeError(
                f"installed server id {installed!r} conflicts with "
                f"requested server id {requested!r}"
            )
        return installed
    server_id = requested or socket.gethostname().split(".", 1)[0].lower()
    if not SERVER_ID_PATTERN.fullmatch(server_id) or server_id == "auto":
        raise RuntimeError("cannot determine a valid deployment server id")
    return server_id


def stop_worker():
    snapshot = worker_snapshot()
    if not snapshot["process_running"]:
        return False
    pid = snapshot["pid"]
    identity = stored_worker_identity()
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + WORKER_LIFECYCLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if identity is None or process_identity(pid) != identity:
            return True
        time.sleep(0.05)
    raise RuntimeError(f"worker did not stop within {WORKER_LIFECYCLE_TIMEOUT_SECONDS:g} seconds")


def start_worker(release_id, server_id):
    binary = releases / release_id / "venv" / "bin" / "gpuops"
    if not release_complete(release_id):
        raise RuntimeError("selected release is incomplete")
    completed = subprocess.run(
        [str(binary), "worker", "start", "--server-id", server_id, "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("worker start failed: " + completed.stdout[-2000:])
    deadline = time.monotonic() + WORKER_LIFECYCLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        snapshot = worker_snapshot(server_id)
        if snapshot["ready"]:
            return snapshot
        time.sleep(0.1)
    raise RuntimeError("worker did not publish a fresh running heartbeat")


def current_release():
    if current.exists() and not current.is_symlink():
        raise RuntimeError("current exists but is not a managed symlink")
    if not current.is_symlink():
        return None
    target = (current.parent / os.readlink(current)).resolve()
    releases_root = releases.resolve()
    if target.parent != releases_root or not RELEASE_ID_PATTERN.fullmatch(target.name):
        raise RuntimeError("current symlink points outside the release directory")
    return target.name


def release_rows():
    rows = []
    if not releases.exists():
        return rows
    for release in releases.iterdir():
        if not release.is_dir() or not RELEASE_ID_PATTERN.fullmatch(release.name):
            continue
        installed_at = None
        metadata_path = release / "release.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            installed_at = metadata.get("installed_at")
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        rows.append(
            {
                "release_id": release.name,
                "installed_at": installed_at,
                "size_bytes": directory_size(release),
            }
        )
    rows.sort(key=lambda row: (row["installed_at"] or "", row["release_id"]), reverse=True)
    return rows


def release_complete(release_id):
    release = releases / release_id
    return (
        (release / "venv" / "bin" / "gpuops").exists()
        and (release / "release.json").exists()
        and not (release / ".installing").exists()
    )


def ensure_rollback_queue_compatible(release_id):
    """Refuse rollback before mutation when its release cannot read durable jobs."""
    jobs = state / "queue" / "jobs"
    if not jobs.exists() or not any(jobs.glob("*.json")):
        return
    python = releases / release_id / "venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError("rollback target has no Python interpreter for compatibility check")
    program = (
        "import sys\n"
        "from pathlib import Path\n"
        "from gpuops.models import JobRecord\n"
        "for path in sorted(Path(sys.argv[1]).glob('*.json')):\n"
        "    try:\n"
        "        JobRecord.model_validate_json(path.read_text(encoding='utf-8'))\n"
        "    except Exception as error:\n"
        "        print(f'{path.name}: {type(error).__name__}: {error}', file=sys.stderr)\n"
        "        raise SystemExit(2)\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", program, str(jobs)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=ROLLBACK_COMPATIBILITY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("rollback compatibility check timed out") from error
    if completed.returncode != 0:
        details = completed.stdout[-2000:].strip()
        suffix = ": " + details if details else ""
        raise RuntimeError(
            "rollback target cannot read persisted GPUOps queue records" + suffix
        )


def status():
    active = current_release()
    rows = release_rows()
    previous = next((row["release_id"] for row in rows if row["release_id"] != active), None)
    installed = active is not None and release_complete(active)
    server_id = installed_server_id()
    state_bytes, share_bytes, cache_bytes = usage()
    worker = worker_snapshot(server_id)
    return {
        "installed": installed,
        "current_release": active,
        "previous_release": previous,
        "releases": rows,
        "share_bytes": share_bytes,
        "budget_bytes": SHARE_BUDGET,
        "cache_bytes": cache_bytes,
        "state_bytes": state_bytes,
        "total_bytes": state_bytes + share_bytes + cache_bytes,
        "total_budget_bytes": TOTAL_BUDGET,
        "worker_ready": worker["ready"],
        "worker_release": release_for_pid(worker["pid"]),
        "executable": str(executable) if executable.is_symlink() else None,
        "server_id": server_id,
    }


def make_private(path):
    for root, directories, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        os.chmod(root_path, 0o700)
        for name in directories:
            candidate = root_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        for name in filenames:
            candidate = root_path / name
            if candidate.is_symlink():
                continue
            mode = candidate.stat().st_mode
            os.chmod(candidate, 0o700 if mode & 0o111 else 0o600)


def ensure_executable_link():
    local_bin.mkdir(parents=True, exist_ok=True, mode=0o700)
    if executable.exists() and not executable.is_symlink():
        raise RuntimeError("refusing to replace an unmanaged ~/.local/bin/gpuops file")
    if executable.is_symlink() and os.readlink(executable) == expected_executable_target:
        return
    atomic_symlink(executable, expected_executable_target)


def prune_releases(keep):
    keep = set(keep) | live_release_ids()
    if not releases.exists():
        return
    for release in releases.iterdir():
        if (
            release.is_dir()
            and RELEASE_ID_PATTERN.fullmatch(release.name)
            and release.name not in keep
        ):
            shutil.rmtree(release)


def projected_share_size(keep, additional_bytes=0):
    total = directory_size(share)
    if not releases.exists():
        return total
    for release in releases.iterdir():
        if (
            release.is_dir()
            and RELEASE_ID_PATTERN.fullmatch(release.name)
            and release.name not in keep
        ):
            total -= directory_size(release)
    return max(0, total) + additional_bytes


def deploy():
    if not isinstance(RELEASE_ID, str) or not RELEASE_ID_PATTERN.fullmatch(RELEASE_ID):
        raise RuntimeError("release id is invalid")
    if not isinstance(DEPLOYMENT_ID, str) or not DEPLOYMENT_ID_PATTERN.fullmatch(DEPLOYMENT_ID):
        raise RuntimeError("deployment id is invalid")
    upload_directory = cache / "uploads" / RELEASE_ID / DEPLOYMENT_ID
    uploads = sorted(
        candidate for candidate in upload_directory.iterdir()
        if candidate.is_file() and ARTIFACT_PATTERN.fullmatch(candidate.name)
    )
    if len(uploads) != 1:
        raise RuntimeError("expected exactly one artifact in this deployment upload directory")
    upload = uploads[0]
    os.chmod(upload, 0o600)
    artifact_name = upload.name
    active_before = current_release()
    target = releases / RELEASE_ID
    changed = active_before != RELEASE_ID
    server_id = deployment_server_id(REQUESTED_SERVER_ID)
    try:
        if not release_complete(RELEASE_ID):
            if target.exists() and target.name != active_before:
                shutil.rmtree(target)
            elif target.exists():
                raise RuntimeError("current release is incomplete")
            if changed:
                prune_releases({value for value in (active_before, RELEASE_ID) if value})
            target.mkdir(mode=0o700)
            installing = target / ".installing"
            installing.write_text("installing\n", encoding="utf-8")
            os.chmod(installing, 0o600)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    str(target / "venv"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                raise RuntimeError("venv creation failed: " + completed.stdout[-2000:])
            ensure_owned_budget()
            python = target / "venv" / "bin" / "python"
            completed = subprocess.run(
                [str(python), "-m", "pip", "install", str(upload)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
            )
            if completed.returncode != 0:
                raise RuntimeError("pip install failed: " + completed.stdout[-4000:])
            ensure_owned_budget()
            atomic_json(
                target / "release.json",
                {
                    "schema": SCHEMA,
                    "release_id": RELEASE_ID,
                    "artifact": artifact_name,
                    "installed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            make_private(target)
            installing.unlink()
            ensure_owned_budget()

        keep = {RELEASE_ID}
        if active_before and active_before != RELEASE_ID:
            keep.add(active_before)
        elif active_before == RELEASE_ID:
            previous = next(
                (row["release_id"] for row in release_rows() if row["release_id"] != RELEASE_ID),
                None,
            )
            if previous:
                keep.add(previous)
        keep |= live_release_ids()
        if projected_share_size(keep) > SHARE_BUDGET:
            if active_before != RELEASE_ID and target.exists():
                shutil.rmtree(target)
            raise RuntimeError("GPUOps share directory would exceed the 128 MiB budget")
        ensure_executable_link()
        try:
            if changed:
                stop_worker()
                atomic_symlink(current, "releases/" + RELEASE_ID)
            elif not worker_snapshot(server_id)["ready"]:
                stop_worker()
            start_worker(RELEASE_ID, server_id)
        except Exception:
            with suppress(Exception):
                stop_worker()
            if active_before is None:
                current.unlink(missing_ok=True)
            else:
                atomic_symlink(current, "releases/" + active_before)
                start_worker(active_before, server_id)
            if (
                active_before != RELEASE_ID
                and target.exists()
                and RELEASE_ID not in live_release_ids()
            ):
                shutil.rmtree(target)
            raise
        keep |= live_release_ids()
        prune_releases(keep)
        ensure_owned_budget()
        identity = {"server_id": server_id}
        if SSH_HOST:
            identity["ssh_host"] = SSH_HOST
        else:
            previous_identity = read_json(host_identity_path)
            if previous_identity and previous_identity.get("ssh_host"):
                identity["ssh_host"] = previous_identity["ssh_host"]
        atomic_json(host_identity_path, identity)
        return changed, "deployed " + RELEASE_ID
    except Exception:
        switched = False
        with suppress(Exception):
            switched = changed and current_release() == RELEASE_ID
        if switched:
            with suppress(Exception):
                stop_worker()
            if active_before is None:
                current.unlink(missing_ok=True)
            else:
                atomic_symlink(current, "releases/" + active_before)
                start_worker(active_before, server_id)
        if active_before != RELEASE_ID and target.exists() and RELEASE_ID not in live_release_ids():
            shutil.rmtree(target)
        raise
    finally:
        if upload_directory.exists():
            shutil.rmtree(upload_directory)
        for parent in (upload_directory.parent, upload_directory.parent.parent):
            with suppress(OSError):
                parent.rmdir()


def rollback():
    active = current_release()
    candidates = [row for row in release_rows() if row["release_id"] != active]
    if active is None:
        raise RuntimeError("GPUOps is not installed")
    if not candidates:
        raise RuntimeError("no previous release is available")
    target = candidates[0]["release_id"]
    if not release_complete(target):
        raise RuntimeError("previous release is incomplete")
    ensure_rollback_queue_compatible(target)
    server_id = deployment_server_id()
    ensure_executable_link()
    try:
        stop_worker()
        ensure_rollback_queue_compatible(target)
        atomic_symlink(current, "releases/" + target)
        start_worker(target, server_id)
        prune_releases({active, target} | live_release_ids())
        ensure_owned_budget()
    except Exception:
        with suppress(Exception):
            stop_worker()
        atomic_symlink(current, "releases/" + active)
        start_worker(active, server_id)
        raise
    return True, "rolled back to " + target


def uninstall():
    changed = (
        current.exists()
        or current.is_symlink()
        or releases.exists()
        or executable.is_symlink()
        or host_identity_path.exists()
    )
    uploads = cache / "uploads"
    if uploads.exists() and any(path.is_file() for path in uploads.rglob("*")):
        raise RuntimeError("cannot uninstall while a deployment upload is active")
    active = current_release()
    server_id = deployment_server_id()
    stop_worker()
    live = live_release_ids()
    if live:
        if active is not None:
            start_worker(active, server_id)
        raise RuntimeError("cannot uninstall while managed job supervisors are running")
    if executable.is_symlink():
        if os.readlink(executable) != expected_executable_target:
            raise RuntimeError("refusing to remove an unmanaged gpuops symlink")
        executable.unlink()
    if current.exists() or current.is_symlink():
        current.unlink()
    if releases.exists():
        shutil.rmtree(releases)
    if cache.exists():
        shutil.rmtree(cache)
    host_identity_path.unlink(missing_ok=True)
    ensure_owned_budget()
    return changed, "uninstalled GPUOps release files"


def run_action():
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    share.mkdir(parents=True, exist_ok=True, mode=0o700)
    releases.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(cache, 0o700)
    os.chmod(share, 0o700)
    os.chmod(releases, 0o700)
    lock_path.touch(mode=0o600, exist_ok=True)
    os.chmod(lock_path, 0o600)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ACTION == "deploy":
            changed, message = deploy()
        elif ACTION == "rollback":
            changed, message = rollback()
        elif ACTION == "uninstall":
            changed, message = uninstall()
        elif ACTION == "status":
            changed, message = False, "status"
        else:
            raise RuntimeError("unsupported deployment action")
        return {
            "ok": True,
            "action": ACTION,
            "changed": changed,
            "message": message,
            "status": status(),
        }


def main():
    try:
        response = run_action()
    except Exception as error:
        response = {
            "ok": False,
            "action": ACTION,
            "error": type(error).__name__ + ": " + str(error),
        }
        print(json.dumps(response, separators=(",", ":")), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
