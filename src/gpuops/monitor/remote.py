"""Python programs sent to remote hosts through standard input."""

# ruff: noqa: E501

from gpuops.config import SchedulerSettings

# These scripts intentionally use only the Python standard library. GPU telemetry is
# sampled by the Worker; clients only read its shared cache.
_CACHE_READER = r"""
import datetime
import json
import os
import stat

CACHE_PATH = os.path.expanduser("~/.local/state/gpuops/gpu-sample.json")
CACHE_READ_LIMIT_BYTES = 2 * 1024 * 1024
MAX_SAMPLE_AGE_SECONDS = MAX_SAMPLE_AGE_SECONDS_DEFAULT
WORKER_GUIDANCE = "install and start the GPUOps Worker to provide GPU telemetry"


class CacheError(Exception):
    def __init__(self, code, message, sampled_at=None):
        super().__init__(message)
        self.code = code
        self.sampled_at = sampled_at


def parse_timestamp(value, field):
    if not isinstance(value, str):
        raise CacheError("telemetry_unavailable", "{} is missing or invalid".format(field))
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise CacheError("telemetry_unavailable", "{} is invalid".format(field))
    if parsed.tzinfo is None:
        raise CacheError("telemetry_unavailable", "{} must include a timezone".format(field))
    return parsed.astimezone(datetime.timezone.utc)


def annotate_stale_gpu(gpu, now):
    if "sampled_at" not in gpu or gpu.get("error") or gpu.get("error_code"):
        return
    try:
        sampled_at = parse_timestamp(gpu.get("sampled_at"), "gpu.sampled_at")
    except CacheError:
        gpu["error"] = "GPU sample timestamp is invalid"
        gpu["error_code"] = "telemetry_stale"
        return
    age = (now - sampled_at).total_seconds()
    if age < 0:
        gpu["error"] = "GPU sample is from the future"
        gpu["error_code"] = "telemetry_stale"
    elif age > MAX_SAMPLE_AGE_SECONDS:
        gpu["error"] = "GPU sample is stale ({:.1f}s old)".format(age)
        gpu["error_code"] = "telemetry_stale"


def read_cache():
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(CACHE_PATH, flags)
    except FileNotFoundError:
        raise CacheError("telemetry_unavailable", "Worker telemetry cache is absent; {}".format(WORKER_GUIDANCE))
    except OSError as exc:
        raise CacheError("telemetry_unavailable", "Worker telemetry cache cannot be opened: {}".format(exc))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CacheError("telemetry_unavailable", "Worker telemetry cache is not a regular file")
        if metadata.st_size > CACHE_READ_LIMIT_BYTES:
            raise CacheError("telemetry_unavailable", "Worker telemetry cache exceeds the read limit")
        chunks = []
        remaining = CACHE_READ_LIMIT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) > CACHE_READ_LIMIT_BYTES:
        raise CacheError("telemetry_unavailable", "Worker telemetry cache exceeds the read limit")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError("telemetry_unavailable", "Worker telemetry cache is malformed: {}".format(exc))
    if not isinstance(document, dict):
        raise CacheError("telemetry_unavailable", "Worker telemetry cache must be a JSON object")
    if "previous" not in document or (
        document["previous"] is not None and not isinstance(document["previous"], dict)
    ):
        raise CacheError("telemetry_unavailable", "Worker telemetry previous snapshot is invalid")
    if "data" not in document or "error_code" not in document:
        raise CacheError("telemetry_unavailable", "Worker telemetry cache is incomplete")

    snapshot = document.get("snapshot")
    if not isinstance(snapshot, dict):
        raise CacheError("telemetry_unavailable", "Worker telemetry cache has no valid snapshot")
    if not isinstance(snapshot.get("server_id"), str) or not snapshot["server_id"]:
        raise CacheError("telemetry_unavailable", "Worker telemetry snapshot has no server_id")
    if not isinstance(snapshot.get("healthy"), bool) or not isinstance(snapshot.get("gpus"), list):
        raise CacheError("telemetry_unavailable", "Worker telemetry snapshot is invalid")
    if any(not isinstance(gpu, dict) for gpu in snapshot["gpus"]):
        raise CacheError("telemetry_unavailable", "Worker telemetry snapshot GPUs are invalid")
    if snapshot.get("error") is not None and not isinstance(snapshot.get("error"), str):
        raise CacheError("telemetry_unavailable", "Worker telemetry snapshot error is invalid")

    sampled_at = parse_timestamp(snapshot.get("sampled_at"), "snapshot.sampled_at")
    now = datetime.datetime.now(datetime.timezone.utc)
    age = (now - sampled_at).total_seconds()
    if age < 0:
        raise CacheError("telemetry_stale", "Worker telemetry sample is from the future", snapshot["sampled_at"])
    if age > MAX_SAMPLE_AGE_SECONDS:
        raise CacheError("telemetry_stale", "Worker telemetry sample is stale ({:.1f}s old)".format(age), snapshot["sampled_at"])

    data = document.get("data")
    error_code = document.get("error_code")
    if error_code is not None and not isinstance(error_code, str):
        raise CacheError("telemetry_unavailable", "Worker telemetry error_code is invalid", snapshot["sampled_at"])
    if not snapshot["healthy"] or snapshot.get("error") or error_code:
        code = error_code if error_code in {"gpu_query_timeout", "gpu_query_failed"} else "telemetry_unavailable"
        message = snapshot.get("error") or "Worker GPU telemetry sample is unhealthy"
        raise CacheError(code, message, snapshot["sampled_at"])
    if not isinstance(data, dict) or not isinstance(data.get("gpus"), list):
        raise CacheError("telemetry_unavailable", "Worker telemetry cache has no valid gpustat data", snapshot["sampled_at"])
    if any(not isinstance(gpu, dict) for gpu in data["gpus"]):
        raise CacheError("telemetry_unavailable", "Worker telemetry gpustat GPUs are invalid", snapshot["sampled_at"])
    collected_at = parse_timestamp(data.get("collected_at"), "data.collected_at")
    if collected_at != sampled_at:
        raise CacheError("telemetry_unavailable", "Worker telemetry timestamps do not match", snapshot["sampled_at"])
    for gpu in data["gpus"]:
        annotate_stale_gpu(gpu, now)
    return document, snapshot, data
"""


COLLECTOR_SCRIPT = _CACHE_READER + r"""
try:
    document, snapshot, data = read_cache()
except CacheError as exc:
    print(json.dumps({"collection_error": {
        "code": exc.code, "message": str(exc)[:500],
    }}, separators=(",", ":")), flush=True)
else:
    print(json.dumps(data, separators=(",", ":"), ensure_ascii=False), flush=True)
"""


DOCTOR_SCRIPT = _CACHE_READER + r"""
import shutil
import sys

result = {
    "python": sys.version.split()[0],
    "gpustat_path": shutil.which("gpustat"),
    "gpustat_ok": False,
    "error": None,
    "error_code": None,
    "source": "worker_cache",
    "sampled_at": None,
    "cache_fresh": False,
}
try:
    document, snapshot, data = read_cache()
except CacheError as exc:
    result["error"] = str(exc)[:500]
    result["error_code"] = exc.code
    result["sampled_at"] = exc.sampled_at
else:
    result["gpustat_ok"] = True
    result["sampled_at"] = snapshot["sampled_at"]
    result["cache_fresh"] = True
print(json.dumps(result, separators=(",", ":")), flush=True)
"""


_CACHE_DEFAULT = (
    f"MAX_SAMPLE_AGE_SECONDS_DEFAULT = {SchedulerSettings().max_snapshot_age_seconds!r}\n"
)
COLLECTOR_SCRIPT = _CACHE_DEFAULT + COLLECTOR_SCRIPT
DOCTOR_SCRIPT = _CACHE_DEFAULT + DOCTOR_SCRIPT
