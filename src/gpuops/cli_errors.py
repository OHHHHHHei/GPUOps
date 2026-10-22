"""Small, additive error contract for human and JSON command output."""

from __future__ import annotations

import json
import shlex
import sys
from typing import Any

from pydantic import ValidationError

from gpuops.control import HostSelectionError, SubmissionConflict


class CommandError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def error_payload(
    error: Exception,
    *,
    operation: str | None = None,
    run_id: str | None = None,
    submission_started: bool = False,
) -> dict[str, Any]:
    code = getattr(error, "code", None)
    if code is None:
        if isinstance(error, ValidationError):
            code = (
                "invalid_manifest"
                if operation in {"submit", "validate"} and not submission_started
                else "invalid_state"
            )
        elif isinstance(error, HostSelectionError):
            code = "host_unavailable"
        elif isinstance(error, SubmissionConflict):
            code = "run_id_conflict"
        elif isinstance(error, OSError):
            code = "io_error"
        elif isinstance(error, ValueError):
            code = "invalid_argument"
        else:
            code = "operation_failed"
    payload: dict[str, Any] = {"ok": False, "error": str(error), "error_code": code}
    identity = getattr(error, "run_id", None) or run_id
    if identity is not None:
        payload["run_id"] = identity
    server_id = getattr(error, "server_id", None)
    if server_id is not None:
        payload["server_id"] = server_id
    state = getattr(error, "submission_state", None)
    if state is not None:
        payload["submission_state"] = state
    elif operation == "submit":
        state = "unknown" if submission_started else "not_accepted"
        if (
            isinstance(error, (HostSelectionError, SubmissionConflict))
            or code == "unsupported_option"
        ):
            state = "not_accepted"
        payload["submission_state"] = state
    if identity and (
        state == "unknown"
        or (state == "not_accepted" and code in {"worker_unavailable", "storage_quota_exceeded"})
    ):
        payload["recovery_command"] = ["gpuops", "reconcile", identity, "--json"]
    return payload


def print_error(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(f"gpuops: {payload['error']}", file=sys.stderr)
    if payload.get("run_id"):
        print(f"run_id: {payload['run_id']}", file=sys.stderr)
    if payload.get("recovery_command"):
        print(
            f"Recover this submission: {shlex.join(payload['recovery_command'])}", file=sys.stderr
        )
