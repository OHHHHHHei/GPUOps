"""Public registration document: generic execution references and follow-up intent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from gpuops.antigravity_delivery import AntigravityAgentAPI
from gpuops.config import select_client_targets
from gpuops.control import SubmissionJournal
from gpuops.models import RunId, StrictModel

_SESSION_ENV = {
    "codex-desktop": "CODEX_THREAD_ID",
    "claude-desktop": "CLAUDE_CODE_SESSION_ID",
    "antigravity": "ANTIGRAVITY_CONVERSATION_ID",
}


class JobReference(StrictModel):
    run_id: RunId
    server_id: str | None = None


class FollowupTarget(StrictModel):
    harness: str | None = None
    session_id: str | None = None
    workspace: str | None = None


class FollowupRequest(StrictModel):
    api_version: Literal["gpuops.followup/v1"] = "gpuops.followup/v1"
    name: str | None = Field(default=None, min_length=1)
    jobs: list[JobReference] = Field(min_length=1)
    target: FollowupTarget = Field(default_factory=FollowupTarget)
    mode: Literal["per_job", "all_terminal"] = "all_terminal"
    action: str = ""
    context: dict[str, str] = Field(default_factory=dict)
    resume: bool = True
    notify: Literal["none", "terminal", "analysis_complete"] = "none"

    @model_validator(mode="after")
    def require_followup_action(self) -> FollowupRequest:
        if self.resume and not self.action.strip():
            raise ValueError("resuming a session requires an action")
        if not self.resume and self.notify == "analysis_complete":
            raise ValueError("analysis_complete notification requires session continuation")
        return self


def _environment_session(harness: str) -> str | None:
    if harness == "claude-desktop" and os.environ.get("CLAUDE_CODE_ENTRYPOINT") != "claude-desktop":
        return None
    variable = _SESSION_ENV.get(harness)
    return (os.environ.get(variable) or None) if variable else None


def detect_origin() -> tuple[str, str] | None:
    candidates = [
        (harness, session_id)
        for harness in _SESSION_ENV
        if (session_id := _environment_session(harness)) is not None
    ]
    if len(candidates) > 1:
        names = ", ".join(harness for harness, _ in candidates)
        raise ValueError(f"follow-up is unbound: ambiguous chat origin ({names}); specify harness")
    return candidates[0] if candidates else None


def _registration_session(harness: str, session_id: str) -> str:
    if harness == "codex-desktop" and session_id:
        try:
            return str(UUID(session_id))
        except ValueError:
            raise ValueError("Codex session_id must be a UUID") from None
    return session_id


def resolve_target(target: FollowupTarget, *, resume: bool = True) -> tuple[str, str, str]:
    harness = target.harness or None
    current_session = None
    if resume:
        if harness is None:
            origin = detect_origin()
            if origin is None:
                raise ValueError(
                    "follow-up is unbound: open the original chat or provide harness and session_id"
                )
            harness, current_session = origin
        else:
            current_session = _environment_session(harness)
    harness = harness or "codex-desktop"
    session_id = _registration_session(harness, target.session_id or current_session or "")
    if resume and not session_id:
        raise ValueError(f"follow-up is unbound: provide the original {harness} session_id")
    if current_session and session_id != _registration_session(harness, current_session):
        raise ValueError(
            f"follow-up is unbound: session_id conflicts with the current {harness} chat"
        )
    if resume and harness == "antigravity":
        workspace = AntigravityAgentAPI().resolve_workspace(session_id, target.workspace)
    else:
        workspace = Path(target.workspace or Path.cwd()).expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise ValueError("workspace must be a project directory")
    return harness, session_id, str(workspace)


def validate_receipt_session(
    harness: str, saved_session: str, explicit_session: str | None = None
) -> None:
    current_session = _environment_session(harness)
    original = (
        "original Codex chat" if harness == "codex-desktop" else f"original {harness} session"
    )
    if current_session is not None and current_session != saved_session:
        raise ValueError(f"use the {original}; the current session differs")
    supplied_session = explicit_session if explicit_session is not None else current_session
    if not saved_session or supplied_session != saved_session:
        raise ValueError(f"use the {original} or provide its ID with --session")


def resolve_job(reference: JobReference) -> tuple[str, str]:
    journal = SubmissionJournal(Path.home() / ".local/state/gpuops/control-journal")
    entry = journal.get(reference.run_id)
    if entry is not None:
        if reference.server_id and reference.server_id != entry.server_id:
            raise ValueError("server_id conflicts with the saved submission destination")
        return entry.server_id, entry.ssh_host
    if not reference.server_id:
        raise ValueError(f"job {reference.run_id!r} has no local journal entry; provide server_id")
    target = select_client_targets([reference.server_id])[0]
    return target.id, target.ssh_host
