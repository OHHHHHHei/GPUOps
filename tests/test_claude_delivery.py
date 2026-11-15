from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock

import pytest

from gpuops.claude_delivery import ClaudeDesktopSocket
from gpuops.followup_delivery import DeliveryUncertain
from gpuops.followups import Followup


@pytest.fixture
def followup(tmp_path: Path) -> Followup:
    return Followup(
        run_id="experiment-1",
        server_id="gpu01",
        ssh_host="lab",
        harness="claude-desktop",
        thread_id="original-session",
        workspace=str(tmp_path),
        action="Read the experiment metrics.",
        message="实验已结束。\nRead metrics.json.",
    )


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    # macOS Unix socket paths must fit in 103 bytes.
    with TemporaryDirectory(dir="/tmp", prefix="gpuops-") as directory:
        yield Path(directory)


def _register(home: Path, inbox: Path, **fields: object) -> None:
    session = {
        "pid": os.getpid(),
        "entrypoint": "claude-desktop",
        "sessionId": "original-session",
        "messagingSocketPath": str(inbox),
        **fields,
    }
    registry = home / ".claude" / "sessions"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / f"{os.getpid()}.json").write_text(json.dumps(session), encoding="utf-8")


def _listen(inbox: Path) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(2)
    listener.bind(str(inbox))
    listener.listen()
    return listener


def _read_message(listener: socket.socket) -> bytes:
    receiver, _ = listener.accept()
    with receiver:
        receiver.settimeout(2)
        with receiver.makefile("rb") as stream:
            return stream.read()


def test_writes_one_utf8_message_to_original_inbox(tmp_path, socket_dir, followup):
    inbox = socket_dir / "inbox.sock"
    _register(tmp_path, inbox)

    with _listen(inbox) as listener:
        result = ClaudeDesktopSocket(home=tmp_path).send(followup)
        payload = _read_message(listener)

    assert payload.endswith(b"\n")
    assert payload.count(b"\n") == 1
    assert "实验已结束。".encode() in payload
    assert json.loads(payload) == {
        "type": "user",
        "session_id": followup.thread_id,
        "message": {"role": "user", "content": followup.message},
    }
    assert result.returncode == 0
    assert result.stdout == "Claude socket write completed; awaiting acknowledgement"


@pytest.mark.parametrize("state", ["missing", "wrong-session", "wrong-entrypoint", "dead-pid"])
def test_missing_or_inactive_session_is_retryable(
    tmp_path, socket_dir, followup, monkeypatch, state
):
    inbox = socket_dir / "other.sock"
    fields = {
        "wrong-session": {"sessionId": "another-session"},
        "wrong-entrypoint": {"entrypoint": "cli"},
    }.get(state, {})
    if state != "missing":
        _register(tmp_path, inbox, **fields)
    if state == "dead-pid":
        monkeypatch.setattr(os, "kill", Mock(side_effect=ProcessLookupError))

    with _listen(inbox), pytest.raises(OSError, match="no active inbox"):
        ClaudeDesktopSocket(home=tmp_path).send(followup)


def test_connection_refused_before_write_is_retryable(tmp_path, socket_dir, followup):
    inbox = socket_dir / "closed.sock"
    _register(tmp_path, inbox)
    _listen(inbox).close()

    with pytest.raises(ConnectionRefusedError):
        ClaudeDesktopSocket(home=tmp_path).send(followup)


def test_write_failure_after_connection_is_uncertain(tmp_path, socket_dir, followup, monkeypatch):
    inbox = socket_dir / "inbox.sock"
    _register(tmp_path, inbox)
    client = MagicMock()
    client.__enter__.return_value = client
    client.sendall.side_effect = TimeoutError

    with _listen(inbox):
        monkeypatch.setattr(socket, "socket", Mock(return_value=client))
        with pytest.raises(DeliveryUncertain, match="verify the original chat"):
            ClaudeDesktopSocket(home=tmp_path).send(followup)
    client.connect.assert_called_once_with(str(inbox))


def test_rereads_registration_after_inbox_replacement(tmp_path, socket_dir, followup):
    adapter = ClaudeDesktopSocket(home=tmp_path)
    original, replacement = socket_dir / "first.sock", socket_dir / "second.sock"
    _register(tmp_path, original)
    with _listen(original) as first:
        adapter.send(followup)
        assert _read_message(first)

    _register(tmp_path, replacement)
    with _listen(replacement) as second:
        adapter.send(followup)
        assert json.loads(_read_message(second))["session_id"] == followup.thread_id
