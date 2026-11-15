"""Deliver a notice to the original Antigravity desktop conversation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

from gpuops.followup_delivery import DeliveryUncertain

if TYPE_CHECKING:
    from gpuops.followups import Followup


_LANGUAGE_SERVER = "/Applications/Antigravity.app/Contents/Resources/bin/language_server"
_TIMEOUT_SECONDS = 30


class AntigravityAgentAPI:
    def __init__(self, home: Path | None = None, executable: Path | str | None = None):
        self.home = home if home is not None else Path.home()
        self.executable = str(executable) if executable is not None else _LANGUAGE_SERVER

    def _environment(self) -> dict[str, str]:
        for path in (self.home / ".gemini/antigravity/daemon").glob("hub_*.json"):
            try:
                hub = json.loads(path.read_bytes())
                if not isinstance(hub, dict):
                    continue
                pid, port, token = hub.get("pid"), hub.get("httpPort"), hub.get("csrfToken")
                if type(pid) is not int or pid <= 0 or type(port) is not int:
                    continue
                if not 0 < port <= 65535 or not isinstance(token, str) or not token:
                    continue
                os.kill(pid, 0)
            except (OSError, ValueError, OverflowError):
                continue
            return {
                **os.environ,
                "ANTIGRAVITY_LS_ADDRESS": f"127.0.0.1:{port}",
                "ANTIGRAVITY_CSRF_TOKEN": token,
            }
        raise OSError("Antigravity Desktop is unavailable; no running hub was found")

    def _run(
        self, arguments: list[str], env: dict[str, str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        command = [self.executable, "agentapi", *arguments]
        try:
            return subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except (OSError, ValueError):
            raise OSError("Antigravity agentapi could not start") from None

    def _metadata(self, session_id: str, env: dict[str, str]) -> dict[str, Any]:
        try:
            result = self._run(["get-conversation-metadata", session_id], env)
            if result.returncode == 0:
                metadata = json.loads(result.stdout)["response"]["conversationMetadata"]["metadata"]
                if isinstance(metadata, dict):
                    return metadata
        except subprocess.TimeoutExpired:
            raise OSError("Antigravity conversation lookup timed out") from None
        except (ValueError, KeyError, TypeError):
            pass
        raise OSError("Antigravity conversation metadata is unavailable")

    def metadata(self, session_id: str) -> dict[str, Any]:
        return self._metadata(session_id, self._environment())

    def resolve_workspace(
        self, session_id: str, explicit_workspace: Path | str | None = None
    ) -> Path:
        return _resolve_workspace(self.metadata(session_id), explicit_workspace)

    def send(self, followup: Followup) -> subprocess.CompletedProcess[str]:
        env = self._environment()
        try:
            workspace = _resolve_workspace(
                self._metadata(followup.thread_id, env), followup.workspace
            )
        except (ValueError, RuntimeError) as error:
            raise OSError(str(error)) from None
        if followup.message is None:
            raise OSError("Antigravity follow-up has no completion message")

        try:
            result = self._run(
                ["send-message", "--title=GPUOps 实验结束", followup.thread_id, followup.message],
                env,
                cwd=workspace,
            )
        except subprocess.TimeoutExpired:
            raise DeliveryUncertain("Antigravity message delivery timed out") from None
        if result.returncode != 0:
            raise DeliveryUncertain(
                f"Antigravity send-message exited with status {result.returncode}"
            )
        try:
            receipt = json.loads(result.stdout)["response"]["sendMessage"]
            if receipt["recipientId"] != followup.thread_id or not isinstance(
                receipt["content"], str
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise DeliveryUncertain("Antigravity returned an invalid delivery receipt") from None
        return subprocess.CompletedProcess(result.args, 0, "Antigravity accepted the message", "")


def _resolve_workspace(metadata: dict[str, Any], explicit_workspace: Path | str | None) -> Path:
    workspace_uris = metadata.get("workspaceUris")
    uris = list(workspace_uris) if isinstance(workspace_uris, list) else []
    workspaces = metadata.get("workspaces")
    if isinstance(workspaces, list):
        uris.extend(
            item.get("workspaceFolderAbsoluteUri") for item in workspaces if isinstance(item, dict)
        )

    roots: list[Path] = []
    for uri in uris:
        if not isinstance(uri, str):
            continue
        location = urlsplit(uri)
        if location.scheme != "file" or location.netloc not in ("", "localhost"):
            continue
        root = Path(unquote(location.path))
        if root.is_absolute() and root.is_dir():
            roots.append(root.resolve())
    if not roots:
        raise OSError("Antigravity conversation has no available local project directory")
    if explicit_workspace is None:
        return roots[0]
    workspace = Path(explicit_workspace).expanduser().resolve(strict=True)
    if not workspace.is_dir() or not any(workspace.is_relative_to(root) for root in roots):
        raise ValueError("workspace must be within the Antigravity conversation's project")
    return workspace
