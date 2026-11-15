from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, call

import pytest

from gpuops.antigravity_delivery import AntigravityAgentAPI
from gpuops.followup_registration import (
    FollowupTarget,
    detect_origin,
    resolve_target,
    validate_receipt_session,
)

_CODEX_SESSION = "123e4567-e89b-42d3-a456-426614174000"


@pytest.fixture(autouse=True)
def project_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)


def test_automatic_origin_requires_claude_desktop_and_keeps_codex_uuid(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "claude-session")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")

    assert detect_origin() is None
    with pytest.raises(ValueError, match="unbound: open the original chat"):
        resolve_target(FollowupTarget())

    monkeypatch.setenv("CODEX_THREAD_ID", _CODEX_SESSION.upper())
    assert resolve_target(FollowupTarget()) == (
        "codex-desktop",
        _CODEX_SESSION,
        str(tmp_path.resolve()),
    )

    monkeypatch.delenv("CODEX_THREAD_ID")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "claude-desktop")
    assert resolve_target(FollowupTarget()) == (
        "claude-desktop",
        "claude-session",
        str(tmp_path.resolve()),
    )


def test_explicit_harness_disambiguates_and_rejects_same_harness_conflict(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_THREAD_ID", _CODEX_SESSION)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "claude-session")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "claude-desktop")

    with pytest.raises(ValueError, match="unbound: ambiguous chat origin"):
        resolve_target(FollowupTarget())

    workspace = str(tmp_path.resolve())
    assert resolve_target(FollowupTarget(harness="claude-desktop")) == (
        "claude-desktop",
        "claude-session",
        workspace,
    )
    assert resolve_target(FollowupTarget(harness="custom-adapter", session_id="custom-id")) == (
        "custom-adapter",
        "custom-id",
        workspace,
    )
    with pytest.raises(ValueError, match="session_id conflicts with the current claude-desktop"):
        resolve_target(FollowupTarget(harness="claude-desktop", session_id="another-chat"))


def test_notification_only_does_not_resolve_session_or_antigravity_project(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_THREAD_ID", _CODEX_SESSION)
    monkeypatch.setenv("ANTIGRAVITY_CONVERSATION_ID", "ag-session")
    metadata = Mock()
    monkeypatch.setattr(AntigravityAgentAPI, "metadata", metadata)
    assert resolve_target(FollowupTarget(), resume=False) == (
        "codex-desktop",
        "",
        str(tmp_path.resolve()),
    )
    assert resolve_target(
        FollowupTarget(harness="antigravity", session_id="other-session"), resume=False
    ) == ("antigravity", "other-session", str(tmp_path.resolve()))
    metadata.assert_not_called()


def test_antigravity_uses_original_project_by_default_from_a_subdirectory(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTIGRAVITY_CONVERSATION_ID", "ag-session")
    project = tmp_path / "project"
    child = project / "experiments"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    metadata = Mock(return_value={"workspaceUris": [project.as_uri()]})
    monkeypatch.setattr(AntigravityAgentAPI, "metadata", metadata)

    assert resolve_target(FollowupTarget()) == ("antigravity", "ag-session", str(project.resolve()))
    assert resolve_target(FollowupTarget(workspace=".")) == (
        "antigravity",
        "ag-session",
        str(child.resolve()),
    )
    assert metadata.call_args_list == [call("ag-session"), call("ag-session")]


def test_receipt_uses_its_harness_environment_even_with_an_explicit_id(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", _CODEX_SESSION)
    monkeypatch.setenv("ANTIGRAVITY_CONVERSATION_ID", "ag-current")

    validate_receipt_session("antigravity", "ag-current")
    with pytest.raises(ValueError, match="the current session differs"):
        validate_receipt_session("antigravity", "ag-original", "ag-original")

    monkeypatch.delenv("ANTIGRAVITY_CONVERSATION_ID")
    validate_receipt_session("antigravity", "ag-original", "ag-original")
    with pytest.raises(ValueError, match="with --session"):
        validate_receipt_session("custom-adapter", _CODEX_SESSION)
    validate_receipt_session("custom-adapter", _CODEX_SESSION, _CODEX_SESSION)
