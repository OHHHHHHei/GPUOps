"""Registration and user-level scheduling for experimental Codex Desktop follow-ups."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import UUID

from gpuops.cli_errors import error_payload, print_error
from gpuops.config import select_client_targets
from gpuops.control import SubmissionJournal
from gpuops.file_io import atomic_write_bytes
from gpuops.followups import CodexQueue, Followup, FollowupStore, check_followups
from gpuops.transport import SshWorkerTransport

LABEL = "io.gpuops.followups"


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("followup", help="experimental original Codex chat follow-ups")
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
    for name in ("show", "stop", "retry", "acknowledge", "complete", "list", "disable", "_check"):
        command = commands.add_parser(name)
        if name in {"show", "stop", "retry", "acknowledge", "complete"}:
            command.add_argument("followup_id")
        if name == "complete":
            command.add_argument("--summary", required=True)
        command.add_argument("--json", action="store_true", dest="as_json")


def enable(root: Path) -> None:
    if sys.platform != "darwin":
        raise ValueError("Codex Desktop follow-up scheduling currently requires macOS")
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
        "codex_home": str(Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()),
    }
    atomic_write_bytes(root / "codex.json", json.dumps(settings).encode())
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    service = f"gui/{os.getuid()}/{LABEL}"
    config = plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [sys.executable, "-m", "gpuops", "followup", "_check"],
            "EnvironmentVariables": {
                # npm's codex entry point needs its sibling node under launchd.
                "PATH": f"{Path(executable).parent}:/opt/homebrew/bin:/usr/local/bin:"
                "/usr/bin:/bin:/usr/sbin:/sbin",
            },
            "RunAtLoad": True,
            "StartInterval": 30,
            "LimitLoadToSessionType": "Aqua",
        }
    )
    loaded = subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0
    if loaded and plist.exists() and plist.read_bytes() == config:
        return
    if loaded:
        subprocess.run(
            ["launchctl", "bootout", service], check=True, capture_output=True, timeout=15
        )
    plist.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(plist, config)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)],
        check=True,
        capture_output=True,
        timeout=15,
    )


def _supports_queue(help_text: str) -> bool:
    return "Usage: codex queue" in help_text and "--thread" in help_text


def register(args: argparse.Namespace, store: FollowupStore, root: Path) -> Followup:
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
    enable(root)
    return store.add(followup)


def run_command(args: argparse.Namespace) -> int:
    root = Path.home() / ".local/state/gpuops/followups"
    store = FollowupStore(root / "records")
    try:
        action = args.followup_command
        if action == "_check":
            settings = json.loads((root / "codex.json").read_bytes())
            asyncio.run(
                check_followups(
                    store, SshWorkerTransport(timeout_seconds=10), CodexQueue(**settings)
                )
            )
            return 0
        if action == "add":
            result = register(args, store, root).model_dump(mode="json")
        elif action == "list":
            followups, invalid = store.scan()
            result = {
                "followups": [
                    row.model_dump(mode="json", exclude={"job", "message"}) for row in followups
                ],
                "invalid_records": invalid,
            }
        elif action == "disable":
            subprocess.run(
                ["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
                capture_output=True,
                timeout=15,
            )
            (Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist").unlink(missing_ok=True)
            result = {"enabled": False}
        elif action in {"acknowledge", "complete"}:
            result = store.acknowledge(
                args.followup_id, summary=getattr(args, "summary", None)
            ).model_dump(mode="json")
        else:
            with store.lock():
                followup = store.get(args.followup_id)
                if action == "retry":
                    if followup.status != "delivery_unknown":
                        raise ValueError(
                            "retry requires delivery_unknown; ordinary failures retry automatically"
                        )
                    if os.environ.get("CODEX_THREAD_ID") != followup.thread_id:
                        raise ValueError("verify delivery and retry from the original Codex chat")
                    followup.status = "ready"
                    followup.error = None
                    store.save(followup)
                if action == "stop":
                    if followup.status not in {"waiting", "ready", "stopped"}:
                        raise ValueError(
                            "delivery has already started; stop cannot recall a queued chat message"
                        )
                    followup.status = "stopped"
                    store.save(followup)
                result = followup.model_dump(mode="json")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0
