from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpuops.antigravity_delivery import AntigravityAgentAPI
from gpuops.followup_delivery import DeliveryUncertain
from gpuops.followups import Followup

_TOKEN = "test-hub-token"


@pytest.fixture
def api(tmp_path: Path) -> AntigravityAgentAPI:
    hub = tmp_path / ".gemini/antigravity/daemon/hub_1234.json"
    hub.parent.mkdir(parents=True)
    hub.write_text(json.dumps({"pid": os.getpid(), "httpPort": 1234, "csrfToken": _TOKEN}))
    return AntigravityAgentAPI(home=tmp_path, executable="/test/language_server")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "research project"
    (root / "experiments").mkdir(parents=True)
    return root


@pytest.fixture
def followup(project: Path) -> Followup:
    return Followup(
        run_id="experiment-1",
        server_id="gpu01",
        ssh_host="lab",
        harness="antigravity",
        thread_id="original-session",
        workspace=str(project / "experiments"),
        action="Read the experiment metrics.",
        message="Read metrics.json.",
    )


def _reply(
    name: str, payload: dict[str, object], stderr: bytes = b""
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        ["agentapi"], 0, json.dumps({"response": {name: payload}}).encode(), stderr
    )


def _metadata_reply(project: Path) -> subprocess.CompletedProcess[bytes]:
    return _reply("conversationMetadata", {"metadata": {"workspaceUris": [project.as_uri()]}})


@pytest.mark.parametrize("source", ["workspaceUris", "workspaces"])
def test_workspace_comes_from_original_conversation(api, project, source, tmp_path, monkeypatch):
    uri = project.as_uri()
    roots = [uri] if source == "workspaceUris" else [{"workspaceFolderAbsoluteUri": uri}]
    run = Mock(return_value=_reply("conversationMetadata", {"metadata": {source: roots}}))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.chdir(project / "experiments")

    assert api.resolve_workspace("original-session") == project.resolve()
    assert api.resolve_workspace("original-session", ".") == (project / "experiments").resolve()
    with pytest.raises(ValueError, match="within the Antigravity conversation's project"):
        api.resolve_workspace("original-session", tmp_path)
    assert run.call_args.args[0] == [
        "/test/language_server",
        "agentapi",
        "get-conversation-metadata",
        "original-session",
    ]


def test_stale_hub_prevents_any_command(api, followup, monkeypatch):
    monkeypatch.setattr(os, "kill", Mock(side_effect=ProcessLookupError))
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(OSError, match="no running hub"):
        api.send(followup)
    run.assert_not_called()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError(_TOKEN), "could not start"),
        (subprocess.TimeoutExpired("agentapi", 30, stderr=_TOKEN.encode()), "lookup timed out"),
    ],
)
def test_failed_preflight_is_retryable(api, followup, monkeypatch, error, message):
    run = Mock(side_effect=error)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(OSError, match=message) as failure:
        api.send(followup)
    assert _TOKEN not in str(failure.value)
    assert run.call_count == 1
    assert run.call_args.args[0][2] == "get-conversation-metadata"


def test_workspace_mismatch_prevents_send(api, followup, project, tmp_path, monkeypatch):
    followup.workspace = str(tmp_path)
    run = Mock(return_value=_metadata_reply(project))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(OSError, match="within the Antigravity conversation's project"):
        api.send(followup)
    assert run.call_count == 1


@pytest.mark.parametrize(
    "unconfirmed",
    [
        subprocess.TimeoutExpired("agentapi", 30),
        subprocess.CompletedProcess(["agentapi"], 2, b"", _TOKEN.encode()),
        subprocess.CompletedProcess(["agentapi"], 0, b"not json", b""),
        _reply("sendMessage", {"recipientId": "another-session", "content": _TOKEN}),
    ],
)
def test_send_without_matching_receipt_is_uncertain(
    api, followup, project, monkeypatch, unconfirmed
):
    run = Mock(side_effect=[_metadata_reply(project), unconfirmed])
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(DeliveryUncertain) as failure:
        api.send(followup)
    assert _TOKEN not in str(failure.value)
    assert run.call_count == 2


def test_send_uses_original_recipient_and_hub_environment(api, followup, project, monkeypatch):
    run = Mock(
        side_effect=[
            _metadata_reply(project),
            _reply(
                "sendMessage",
                {"recipientId": followup.thread_id, "content": _TOKEN},
                _TOKEN.encode(),
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)

    result = api.send(followup)

    args = run.call_args.args[0]
    assert args == [
        "/test/language_server",
        "agentapi",
        "send-message",
        "--title=GPUOps 实验结束",
        followup.thread_id,
        followup.message,
    ]
    for invocation in run.call_args_list:
        env = invocation.kwargs["env"]
        assert env["ANTIGRAVITY_LS_ADDRESS"] == "127.0.0.1:1234"
        assert env["ANTIGRAVITY_CSRF_TOKEN"] == _TOKEN
        assert _TOKEN not in invocation.args[0]
    assert run.call_args.kwargs["cwd"] == Path(followup.workspace).resolve()
    assert result.returncode == 0
    assert result.stdout == "Antigravity accepted the message"
    assert result.stderr == ""
