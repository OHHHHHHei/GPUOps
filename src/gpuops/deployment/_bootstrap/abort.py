import fcntl
import json
import os
import re
import shutil
import sys
from contextlib import suppress
from pathlib import Path

os.umask(0o077)
release_id = sys.argv[1] if len(sys.argv) > 1 else ""
deployment_id = sys.argv[2] if len(sys.argv) > 2 else ""
if (
    not re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", release_id)
    or not re.fullmatch(r"[0-9a-f]{32}", deployment_id)
):
    raise RuntimeError("deployment metadata is invalid")
home = Path.home()
share = home / ".local" / "share" / "gpuops"
upload = home / ".cache" / "gpuops" / "uploads" / release_id / deployment_id
share.mkdir(parents=True, exist_ok=True, mode=0o700)
lock_path = share / "deploy.lock"
lock_path.touch(mode=0o600, exist_ok=True)
with lock_path.open("r+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if upload.exists():
        shutil.rmtree(upload)
    for parent in (upload.parent, upload.parent.parent):
        with suppress(OSError):
            parent.rmdir()
print(json.dumps({"ok": True, "action": "abort"}, separators=(",", ":")))
