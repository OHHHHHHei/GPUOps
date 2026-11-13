"""Public registration document: generic execution references and follow-up intent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from gpuops.config import select_client_targets
from gpuops.control import SubmissionJournal
from gpuops.models import RunId, StrictModel


class JobReference(StrictModel):
    run_id: RunId
    server_id: str | None = None


class FollowupTarget(StrictModel):
    harness: str = "codex-desktop"
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


def resolve_target(target: FollowupTarget, *, resume: bool = True) -> tuple[str, str]:
    session_id = target.session_id or (
        os.environ.get("CODEX_THREAD_ID") if target.harness == "codex-desktop" else None
    )
    if resume and not session_id:
        raise ValueError("provide the original Harness session_id (Codex infers CODEX_THREAD_ID)")
    if session_id and target.harness == "codex-desktop":
        session_id = str(UUID(session_id))
    workspace = Path(target.workspace or Path.cwd()).expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace must be a project directory")
    return session_id or "", str(workspace)


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
