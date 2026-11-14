"""CLI interfaces and user-level scheduling for session follow-ups."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from gpuops.cli_errors import error_payload, print_error
from gpuops.config import select_client_targets
from gpuops.control import SubmissionJournal
from gpuops.file_io import atomic_write_bytes
from gpuops.followups import Followup, FollowupStore, pending_email

if TYPE_CHECKING:
    from gpuops.background import Background


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "followup", help="completion events, groups, and original-session follow-ups"
    )
    commands = parser.add_subparsers(dest="followup_command", required=True)
    add = commands.add_parser("add", help="follow one submitted job in the current Codex chat")
    add.add_argument("run_id")
    add.add_argument(
        "--action", required=True, help="what the original Agent should do after completion"
    )
    add.add_argument("--server", help="needed only if this client has no submission journal entry")
    add.add_argument("--thread", default=None, help="defaults to CODEX_THREAD_ID")
    add.add_argument("--workspace", type=Path, default=None)
    add.add_argument("--json", action="store_true", dest="as_json")
    register_command = commands.add_parser("register", help="register a task group from JSON")
    register_command.add_argument("file", help="gpuops.followup/v1 JSON, or - for stdin")
    register_command.add_argument("--json", action="store_true", dest="as_json")
    configure = commands.add_parser(
        "configure", help="configure follow-up email once on this client"
    )
    configure.add_argument("--email-config", type=Path, required=True)
    configure.add_argument("--json", action="store_true", dest="as_json")
    adapter = commands.add_parser("adapter", help="configure an external Harness delivery adapter")
    adapter.add_argument("name")
    adapter.add_argument("file", type=Path)
    adapter.add_argument("--json", action="store_true", dest="as_json")
    for name in (
        "show",
        "stop",
        "retry",
        "acknowledge",
        "complete",
        "list",
        "enable",
        "disable",
        "check",
        "capabilities",
        "schema",
    ):
        command = commands.add_parser(name)
        if name in {"show", "stop", "retry", "acknowledge", "complete"}:
            command.add_argument("followup_id")
        if name == "complete":
            command.add_argument("--summary", required=True)
        if name in {"acknowledge", "complete", "retry"}:
            command.add_argument("--session", help="original session ID for an external Harness")
        command.add_argument("--json", action="store_true", dest="as_json")


def prepare_delivery(root: Path, *, harness: str | None = "codex-desktop") -> None:
    if sys.platform != "darwin":
        raise ValueError("automatic follow-up scheduling currently requires macOS")
    if harness == "codex-desktop":
        executable = shutil.which("codex")
        if executable is None:
            raise ValueError("install a Codex CLI with the queue command first")
        # Codex prints its top-level help with status 0 for unknown subcommands.
        result = subprocess.run(
            [executable, "queue", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode or not _supports_queue(result.stdout):
            raise ValueError("the installed Codex CLI does not support queue")
        settings = {
            "executable": executable,
            "codex_home": str(
                Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()
            ),
        }
        atomic_write_bytes(root / "codex.json", json.dumps(settings).encode())
    elif harness is not None:
        from gpuops.followup_delivery import load_adapters

        if harness not in load_adapters():
            raise ValueError(f"configure the {harness!r} Harness adapter before registering")


def _supports_queue(help_text: str) -> bool:
    return "Usage: codex queue" in help_text and "--thread" in help_text


def register(args: argparse.Namespace, store: FollowupStore, background: Background) -> Followup:
    thread_id = args.thread or os.environ.get("CODEX_THREAD_ID")
    if not thread_id:
        raise ValueError("run from the original Codex chat, or provide its --thread ID")
    thread_id = str(UUID(thread_id))
    workspace = (args.workspace or Path.cwd()).expanduser().resolve(strict=True)
    if not workspace.is_dir() or not args.action.strip():
        raise ValueError("a project directory and nonempty --action are required")
    journal = SubmissionJournal(Path.home() / ".local/state/gpuops/control-journal")
    entry = journal.get(args.run_id)
    if entry is not None:
        if args.server and args.server != entry.server_id:
            raise ValueError("--server conflicts with the saved submission destination")
        server_id, ssh_host = entry.server_id, entry.ssh_host
    elif args.server:
        target = select_client_targets([args.server])[0]
        server_id, ssh_host = target.id, target.ssh_host
    else:
        raise ValueError("job is absent from the local submission journal; provide --server")
    followup = Followup(
        run_id=args.run_id,
        server_id=server_id,
        ssh_host=ssh_host,
        thread_id=thread_id,
        workspace=str(workspace),
        action=args.action,
    )
    prepare_delivery(background.followup_root)
    return store.add(followup, on_registered=lambda: background.enable_followups_locked(store))


def run_command(args: argparse.Namespace) -> int:
    from gpuops.background import PAUSED_MESSAGE, Background

    background = Background()
    store = background.event_store()
    root = background.followup_root
    result: dict[str, Any]
    try:
        action = args.followup_command
        if action == "check":
            result = asyncio.run(background.run_round(followup_only=True))
            if result.get("reason") == "followups_paused":
                result["message"] = PAUSED_MESSAGE
        elif action == "add":
            result = register(args, store, background).model_dump(mode="json")
        elif action == "register":
            result = register_group(args.file, store, background)
        elif action == "adapter":
            with store.lock():
                result = configure_or_describe(args)
                if background.followups_enabled():
                    background.sync_locked(store)
        elif action in {"configure", "capabilities", "schema"}:
            result = configure_or_describe(args)
        elif action == "list":
            from gpuops.followup_groups import GroupStore, summarize_group

            followups, invalid = store.scan()
            groups, invalid_groups = GroupStore(root / "groups").scan()
            result = {
                "followups": [
                    row.model_dump(mode="json", exclude={"job", "message"}) for row in followups
                ],
                "groups": [summarize_group(group, store) for group in groups],
                "invalid_records": [*invalid, *invalid_groups],
            }
        elif action in {"enable", "disable"}:
            enabled = action == "enable"
            if enabled and sys.platform != "darwin":
                raise ValueError("automatic follow-up scheduling currently requires macOS")
            with store.lock():
                background.set_switch_locked("followups", enabled)
                background.sync_locked(store)
            result = {"enabled": enabled}
        elif action in {"acknowledge", "complete"}:
            followup = store.acknowledge(
                args.followup_id, summary=getattr(args, "summary", None), session_id=args.session
            )
            result = followup.model_dump(mode="json")
            if (
                action == "complete"
                and followup.notify == "analysis_complete"
                and pending_email(followup)
            ):
                with store.lock():
                    _schedule_existing_work(background, store, result)
        elif action in {"show", "stop", "retry"}:
            from gpuops.followup_groups import GroupStore, stop_group, summarize_group

            group_store = GroupStore(root / "groups")
            if (
                action != "retry"
                and (group_store.directory / f"{UUID(args.followup_id)}.json").exists()
            ):
                result = (
                    stop_group(group_store, store, args.followup_id)
                    if action == "stop"
                    else summarize_group(group_store.get(args.followup_id), store)
                )
                if action == "stop":
                    with store.lock():
                        background.sync_locked(store)
                print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
                return 0
            with store.lock():
                followup = store.get(args.followup_id)
                if action == "retry":
                    if followup.status != "delivery_unknown":
                        raise ValueError(
                            "retry requires delivery_unknown; ordinary failures retry automatically"
                        )
                    if (args.session or os.environ.get("CODEX_THREAD_ID")) != followup.thread_id:
                        raise ValueError(
                            "verify delivery and retry from the original Harness session"
                        )
                    followup.status = "ready"
                    followup.error = None
                    store.save(followup)
                if action == "stop":
                    pending_email_only = (
                        followup.status == "observed" and followup.email_sent_at is None
                    )
                    if (
                        followup.status not in {"waiting", "ready", "stopped"}
                        and not pending_email_only
                    ):
                        raise ValueError(
                            "delivery has already started; stop cannot recall a queued chat message"
                        )
                    followup.status = "stopped"
                    store.save(followup)
                result = followup.model_dump(mode="json")
                if action == "retry":
                    _schedule_existing_work(background, store, result)
                elif action == "stop":
                    background.sync_locked(store)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


def _schedule_existing_work(
    background: Background, store: FollowupStore, result: dict[str, Any]
) -> None:
    from gpuops.background import PAUSED_MESSAGE

    if background.followups_enabled():
        background.sync_locked(store)
    else:
        result["background_message"] = PAUSED_MESSAGE


def register_group(file: str, store: FollowupStore, background: Background) -> dict[str, Any]:
    from gpuops.followup_delivery import read_email_settings
    from gpuops.followup_groups import GroupStore, RemoteJob, create_group, summarize_group
    from gpuops.followup_registration import FollowupRequest, resolve_job, resolve_target

    raw = sys.stdin.read() if file == "-" else Path(file).expanduser().read_text()
    request = FollowupRequest.model_validate_json(raw)
    thread_id, workspace = resolve_target(request.target, resume=request.resume)
    members = []
    for reference in request.jobs:
        server_id, ssh_host = resolve_job(reference)
        members.append(RemoteJob(run_id=reference.run_id, server_id=server_id, ssh_host=ssh_host))
    if request.notify != "none" and read_email_settings() is None:
        raise ValueError("configure follow-up email before registering an email notification")
    prepare_delivery(
        background.followup_root, harness=request.target.harness if request.resume else None
    )
    group = create_group(
        GroupStore(background.followup_root / "groups"),
        store,
        members,
        thread_id=thread_id,
        workspace=workspace,
        action=request.action,
        mode=request.mode,
        harness=request.target.harness,
        notify=request.notify,
        resume=request.resume,
        context=request.context,
        name=request.name,
        on_registered=lambda: background.enable_followups_locked(store),
    )
    return summarize_group(group, store)


def configure_or_describe(args: argparse.Namespace) -> dict[str, Any]:
    from gpuops.followup_delivery import (
        adapter_capabilities,
        configure_adapter,
        configure_email,
        read_email_settings,
    )
    from gpuops.followup_registration import FollowupRequest

    if args.followup_command == "schema":
        return {"schema": FollowupRequest.model_json_schema()}
    if args.followup_command == "configure":
        email = configure_email(args.email_config)
        return {"email_configured": True, "email_recipient": email.recipient}
    if args.followup_command == "adapter":
        adapter = configure_adapter(args.name, args.file)
        return {"adapter": args.name, "idempotent": adapter.idempotent}
    settings = read_email_settings()
    return {
        "api_version": "gpuops.followup/v1",
        "event_api_version": "gpuops.completion/v1",
        "modes": ["per_job", "all_terminal"],
        "notification_phases": ["none", "terminal", "analysis_complete"],
        "adapters": adapter_capabilities(),
        "email_configured": settings is not None,
        "background_platform": "macos",
        "codex_unloaded_session_wakeup": False,
    }
