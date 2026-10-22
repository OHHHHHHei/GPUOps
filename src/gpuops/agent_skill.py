"""Optional Codex guidance, independent of server installation and scheduling."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from gpuops.config import select_client_targets
from gpuops.execution import HostClient

_INSTALL_SKILL = """
import json, os, pathlib, sys, tempfile
root = pathlib.Path(os.environ.get('CODEX_HOME') or pathlib.Path.home() / '.codex')
root = root.expanduser() / 'skills/gpu-infra'
for relative, text in json.load(sys.stdin).items():
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        try:
            handle.write(text)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
print(json.dumps({'installed': str(root)}))
"""


async def install_skill(server_id: str | None = None) -> dict[str, str]:
    from gpuops.bundled import agent_files

    files = agent_files()
    if server_id is not None:
        target = select_client_targets([server_id])[0]
        result = await HostClient().run_command(
            target.ssh_host,
            ["python3", "-c", _INSTALL_SKILL],
            stdin=json.dumps(files, ensure_ascii=False).encode("utf-8"),
            timeout_seconds=30,
        )
        if result.returncode:
            raise RuntimeError(
                f"skill installation failed: {result.stderr.strip() or result.stdout}"
            )
        return {"server_id": target.id, **json.loads(result.stdout)}
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    root = root / "skills/gpu-infra"
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(text)
                handle.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return {"installed": str(root)}
