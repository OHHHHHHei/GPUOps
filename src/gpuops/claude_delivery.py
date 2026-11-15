"""Deliver a notice to the original Claude Desktop conversation."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from gpuops.followup_delivery import DeliveryUncertain

if TYPE_CHECKING:
    from gpuops.followups import Followup


class ClaudeDesktopSocket:
    def __init__(self, home: Path | None = None):
        self.home = home if home is not None else Path.home()

    def _inbox(self, session_id: str) -> Path:
        for registration in (self.home / ".claude" / "sessions").glob("*.json"):
            try:
                session = json.loads(registration.read_bytes())
                if not isinstance(session, dict) or (
                    session.get("sessionId") != session_id
                    or session.get("entrypoint") != "claude-desktop"
                ):
                    continue
                pid = session.get("pid")
                socket_path = session.get("messagingSocketPath")
                if type(pid) is not int or pid <= 0 or not isinstance(socket_path, str):
                    continue
                os.kill(pid, 0)
                inbox = Path(socket_path)
                if inbox.is_absolute() and inbox.is_socket():
                    return inbox
            except (OSError, ValueError, OverflowError):
                continue
        raise OSError("Claude Desktop session has no active inbox")

    def send(self, followup: Followup) -> subprocess.CompletedProcess[str]:
        if not followup.message:
            raise OSError("Claude Desktop follow-up has no completion message")
        inbox = self._inbox(followup.thread_id)
        frame = {
            "type": "user",
            "session_id": followup.thread_id,
            "message": {"role": "user", "content": followup.message},
        }
        try:
            payload = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        except UnicodeEncodeError:
            raise OSError("Claude Desktop completion message is not valid UTF-8") from None

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(str(inbox))
            try:
                client.sendall(payload)
            except OSError:
                raise DeliveryUncertain(
                    "Claude Desktop socket write failed; verify the original chat before retrying"
                ) from None

        # The inbox does not acknowledge messages on this connection.
        return subprocess.CompletedProcess(
            ["claude-desktop", "socket"],
            0,
            "Claude socket write completed; awaiting acknowledgement",
            "",
        )
