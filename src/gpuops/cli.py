"""Command-line entry point for GPUOps."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gpuops.cli_errors import CommandError, error_payload, print_error
from gpuops.config import MONITOR_TIMEOUT_SECONDS, select_client_targets, select_targets
from gpuops.contracts import validate_run_id
from gpuops.control import SubmissionResult
from gpuops.models import ACTIVE_JOB_STATUSES, JobSpec, validate_new_submission
from gpuops.monitor import MonitorService, ServerTarget
from gpuops.monitor.formatting import format_doctor_table, format_status_table


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CommandError("invalid_argument", message)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="gpuops", description="Local GPU compute control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="show release and interface versions")
    version.add_argument("--json", action="store_true", dest="as_json")

    skill = subparsers.add_parser("skill", help="optional Codex skill installation")
    skill_commands = skill.add_subparsers(dest="skill_command", required=True)
    install = skill_commands.add_parser("install", help="install the bundled Codex skill")
    install.add_argument("--server", help="install on a registered server instead of this machine")
    install.add_argument("--json", action="store_true", dest="as_json")

    server = subparsers.add_parser("server", help="connect SSH servers to GPUOps")
    server_commands = server.add_subparsers(dest="server_command", required=True)
    add = server_commands.add_parser("add", help="install or reuse a Worker via an SSH alias")
    add.add_argument("ssh_alias")
    add.add_argument("--project-root", type=Path, default=None, help=argparse.SUPPRESS)
    add.add_argument("--json", action="store_true", dest="as_json")
    listing = server_commands.add_parser("list", help="list registered servers")
    listing.add_argument("--json", action="store_true", dest="as_json")
    remove = server_commands.add_parser("remove", help="disconnect a server with no active jobs")
    remove.add_argument("server_id")
    remove.add_argument("--json", action="store_true", dest="as_json")

    status = subparsers.add_parser("status", help="show the current GPU state")
    _add_remote_options(status)
    status.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON",
    )

    doctor = subparsers.add_parser("doctor", help="check server access and shared GPU sampling")
    _add_remote_options(doctor)
    doctor.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON",
    )

    validate = subparsers.add_parser("validate", help="validate a gpuops/v1 JSON job manifest")
    validate.add_argument("job_file", help="JSON manifest path, or - for standard input")
    validate.add_argument("--json", action="store_true", dest="as_json")

    web = subparsers.add_parser("web", help="start the local web console")
    web.add_argument(
        "--bind", "--host", dest="host", default="127.0.0.1", help="local bind address"
    )
    web.add_argument("--port", default=8787, type=int, help="local bind port")
    web.add_argument(
        "--timeout",
        default=MONITOR_TIMEOUT_SECONDS,
        type=float,
        help="per-server timeout in seconds",
    )
    web.add_argument("--demo", action="store_true", help="show deterministic demo data")

    worker = subparsers.add_parser("worker", help="manage the host-local remote worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    for name in ("run", "start"):
        command = worker_commands.add_parser(name)
        command.add_argument("--server-id", required=True)
        command.add_argument("--state-dir", type=Path, default=None)
        command.add_argument("--interval", type=float, default=5.0)
        command.add_argument("--json", action="store_true", dest="as_json")
        if name == "run":
            command.add_argument("--once", action="store_true")
    for name in ("stop", "status"):
        command = worker_commands.add_parser(name)
        command.add_argument("--state-dir", type=Path, default=None)
        command.add_argument("--json", action="store_true", dest="as_json")

    remote = subparsers.add_parser("remote", help="serve one fixed worker JSON request")
    remote.add_argument("--server-id", required=True)
    remote.add_argument("--state-dir", type=Path, default=None)

    deploy = subparsers.add_parser("deploy", help="manage user-level remote installations")
    deploy_commands = deploy.add_subparsers(dest="deploy_command", required=True)
    for name in ("install", "status", "rollback", "uninstall"):
        command = deploy_commands.add_parser(name)
        placement = command.add_mutually_exclusive_group()
        placement.add_argument("--server", action="append")
        placement.add_argument("--host", action="append", default=[], help=argparse.SUPPRESS)
        command.add_argument("--json", action="store_true", dest="as_json")
        if name == "install":
            command.add_argument("--project-root", type=Path, default=None)

    submit = subparsers.add_parser("submit", help="durably submit a job to one remote worker")
    submit.add_argument("job_file")
    submit.add_argument("--run-id")
    submit.add_argument("--host", help=argparse.SUPPRESS)
    submit.add_argument("--json", action="store_true", dest="as_json")

    queue = subparsers.add_parser("queue", help="show active job summaries")
    queue.add_argument("--server", action="append")
    queue.add_argument("--json", action="store_true", dest="as_json")
    queue.add_argument("--all", action="store_true", help="return all full records (legacy output)")

    history = subparsers.add_parser("history", help="show recently completed jobs")
    history.add_argument("--server", action="append")
    history.add_argument("--project", help="filter by exact project name")
    history.add_argument("--limit", type=int, default=50, help="page size, 1–100 (default: 50)")
    history.add_argument("--cursor", help="next_cursor from the previous history response")
    history.add_argument("--json", action="store_true", dest="as_json")

    job = subparsers.add_parser("job", help="show one managed job")
    job.add_argument("run_id")
    job.add_argument("--server", "--host", dest="host")
    job.add_argument("--json", action="store_true", dest="as_json")

    logs = subparsers.add_parser("logs", help="show bounded console output for one job")
    logs.add_argument("run_id")
    logs.add_argument("--server", "--host", dest="host")
    logs.add_argument("--tail-bytes", type=int, default=64 * 1024)
    logs.add_argument("--json", action="store_true", dest="as_json")

    cancel = subparsers.add_parser("cancel", help="cancel one managed job safely")
    cancel.add_argument("run_id")
    cancel.add_argument("--server", "--host", dest="host")
    cancel.add_argument("--reason")
    cancel.add_argument("--json", action="store_true", dest="as_json")

    reconcile = subparsers.add_parser("reconcile", help="retry pending submissions safely")
    reconcile.add_argument("run_id", nargs="?", help="recover one saved submission")
    reconcile.add_argument(
        "--list",
        action="store_true",
        dest="list_pending",
        help="list pending local submissions without contacting servers",
    )
    reconcile.add_argument("--json", action="store_true", dest="as_json")

    fleet = subparsers.add_parser("fleet", help="manage workers on selected servers")
    fleet.add_argument("fleet_command", choices=["start", "stop", "status"])
    fleet.add_argument("--server", action="append")
    fleet.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _add_remote_options(parser: argparse.ArgumentParser) -> None:
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--server", action="append")
    placement.add_argument(
        "--host",
        action="append",
        default=[],
        metavar="ID=SSH_ALIAS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=MONITOR_TIMEOUT_SECONDS,
        help="per-server timeout in seconds",
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(arguments)
        if args.command == "reconcile" and args.list_pending and args.run_id:
            raise CommandError(
                "invalid_argument", "reconcile --list cannot be combined with RUN_ID"
            )
    except CommandError as error:
        print_error(error_payload(error), as_json="--json" in arguments)
        return 2
    if args.command == "version":
        from gpuops.version import version_info

        value = version_info()
        print(json.dumps(value) if args.as_json else f"GPUOps {value['version']}")
        return 0
    if args.command == "skill":
        return asyncio.run(_run_skill_command(args))
    if args.command == "server":
        return asyncio.run(_run_server_command(args))
    if args.command == "worker":
        return _run_worker_command(args)
    if args.command == "remote":
        return _run_worker_rpc(args)
    if args.command == "deploy":
        return asyncio.run(_run_deploy_command(args))
    if args.command in {
        "submit",
        "queue",
        "history",
        "job",
        "logs",
        "cancel",
        "reconcile",
        "fleet",
    }:
        return asyncio.run(_run_control_command(args))
    if args.command == "web":
        return _run_web(args.host, args.port, args.demo, args.timeout)
    if args.command == "validate":
        return _validate_job(args.job_file, args.as_json)
    try:
        targets = _parse_targets(args.host) if args.host else select_client_targets(args.server)
    except (OSError, ValueError) as exc:
        print_error(error_payload(exc), as_json=args.as_json)
        return 2
    if args.timeout <= 0:
        print_error(
            error_payload(CommandError("invalid_argument", "--timeout must be positive")),
            as_json=args.as_json,
        )
        return 2
    try:
        return asyncio.run(_run_remote_command(args, targets))
    except (OSError, RuntimeError, ValueError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1


def _run_worker_rpc(args: argparse.Namespace) -> int:
    from gpuops.worker.app import DEFAULT_STATE_DIRECTORY
    from gpuops.worker.rpc import WorkerRpc

    state_directory = args.state_dir or DEFAULT_STATE_DIRECTORY
    response = WorkerRpc(state_directory, server_id=args.server_id).handle_json(
        sys.stdin.buffer.read()
    )
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return 0


async def _run_deploy_command(args: argparse.Namespace) -> int:
    from gpuops.deploy import RemoteDeploymentManager

    try:
        targets = select_targets(args.server) if not args.host else ()
        hosts = args.host or [target.ssh_host for target in targets]
    except (OSError, ValueError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1
    by_host = {target.ssh_host: target.id for target in targets}
    manager = RemoteDeploymentManager()

    async def execute(host: str):  # type: ignore[no-untyped-def]
        try:
            if args.deploy_command == "install":
                result = await manager.deploy(host, args.project_root, server_id=by_host.get(host))
                return {"host": host, "ok": True, "result": asdict(result)}
            if args.deploy_command == "status":
                result = await manager.status(host)
                return {"host": host, "ok": True, "result": asdict(result)}
            if args.deploy_command == "rollback":
                result = await manager.rollback(host)
                return {"host": host, "ok": True, "result": asdict(result)}
            result = await manager.uninstall(host)
            return {"host": host, "ok": True, "result": asdict(result)}
        except (OSError, RuntimeError, ValueError) as error:
            return {"host": host, **error_payload(error)}

    results = await asyncio.gather(*(execute(host) for host in hosts))
    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            if result["ok"]:
                status = result["result"].get("status", result["result"])
                current = status.get("current_release") or "not installed"
                print(f"{result['host']}: {current}")
            else:
                print(f"{result['host']}: {result['error']}", file=sys.stderr)
    return 0 if all(result["ok"] for result in results) else 1


async def _run_server_command(args: argparse.Namespace) -> int:
    from gpuops.servers import add_server, remove_server

    try:
        if args.server_command == "add":
            payload = await add_server(args.ssh_alias, project_root=args.project_root)
        elif args.server_command == "remove":
            payload = await remove_server(args.server_id)
        else:
            payload = {"servers": [asdict(target) for target in select_targets()]}
    except (OSError, RuntimeError, ValueError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False, indent=2))
    return 0


async def _run_skill_command(args: argparse.Namespace) -> int:
    from gpuops.agent_skill import install_skill

    try:
        result = await install_skill(args.server)
    except (OSError, RuntimeError, ValueError) as error:
        print_error(error_payload(error), as_json=args.as_json)
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


async def _run_control_command(args: argparse.Namespace) -> int:
    from gpuops.control import (
        ControlClient,
        RemoteTarget,
        SubmissionJournal,
        generate_run_id,
    )
    from gpuops.models import utc_now
    from gpuops.transport import SshWorkerTransport

    outcome = "success"
    run_id = getattr(args, "run_id", None)
    submission_started = False

    try:
        if args.command in {"submit", "reconcile"}:
            journal = SubmissionJournal(Path("~/.local/state/gpuops/control-journal").expanduser())
        if args.command == "reconcile" and args.list_pending:
            payload = {
                "ok": True,
                "pending": [
                    {
                        "run_id": entry.run_id,
                        "server_id": entry.server_id,
                        "job_id": entry.request_spec.id,
                        "project": entry.request_spec.project,
                        "created_at": entry.created_at.isoformat(),
                    }
                    for entry in journal.list_pending()
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        transport = SshWorkerTransport()
        if args.command == "submit":
            raw = sys.stdin.read() if args.job_file == "-" else Path(args.job_file).read_text()
            spec = JobSpec.model_validate_json(raw)
            if args.host is not None:
                resources = spec.resources.model_copy(update={"host": args.host})
                spec = spec.model_copy(update={"resources": resources})
            run_id = args.run_id or generate_run_id(spec, utc_now())
            validate_run_id(run_id)
            server_ids = None if spec.resources.host == "auto" else [spec.resources.host]
            targets = tuple(
                RemoteTarget(target.id, target.ssh_host)
                for target in select_client_targets(server_ids)
            )
            client = ControlClient(targets, transport, journal)
            submission_started = True
            submitted = await client.submit(spec, run_id=run_id)
            payload = _submission_payload(submitted)
        elif args.command == "reconcile":
            client = ControlClient(
                [RemoteTarget(target.id, target.ssh_host) for target in select_targets()],
                transport,
                journal,
            )
            if args.run_id:
                payload = _submission_payload(await client.reconcile(args.run_id))
            else:
                report = await client.reconcile_pending()
                outcome = _aggregate_outcome(len(report.accepted), len(report.pending))
                payload = {
                    "outcome": outcome,
                    "accepted": list(report.accepted),
                    "pending": list(report.pending),
                    "errors": report.errors,
                    "submission_states": report.outcomes,
                    "error_codes": report.error_codes,
                    "journal_warnings": report.warnings,
                }
        else:
            server_ids = getattr(args, "server", None)
            if args.command in {"job", "logs", "cancel"} and args.host:
                server_ids = [args.host]
            targets = tuple(
                RemoteTarget(target.id, target.ssh_host)
                for target in select_client_targets(server_ids)
            )
        if args.command == "queue":
            rows = await _collect_queues(transport, targets, full=args.all)
            outcome = _aggregate_outcome(
                sum(1 for row in rows if row["online"]),
                sum(1 for row in rows if not row["online"]),
            )
            payload = {"outcome": outcome, "view": "all" if args.all else "active", "servers": rows}
        elif args.command == "history":
            from gpuops.queries import collect_history

            page = await collect_history(
                transport, targets, limit=args.limit, project=args.project, cursor=args.cursor
            )
            rows = [
                {
                    "server_id": host.target.server_id,
                    "online": host.error is None,
                    **(error_payload(host.error) if host.error is not None else {}),
                }
                for host in page.hosts
            ]
            outcome = _aggregate_outcome(
                sum(1 for row in rows if row["online"]),
                sum(1 for row in rows if not row["online"]),
            )
            payload = {
                "outcome": outcome,
                "jobs": [
                    {"server_id": target.server_id, **job.model_dump(mode="json")}
                    for target, job in page.items
                ],
                "next_cursor": page.next_cursor,
                "projects": page.projects,
                "servers": rows,
            }
        elif args.command == "fleet":
            rows = await _fleet_action(transport, targets, args.fleet_command)
            outcome = _aggregate_outcome(
                sum(1 for row in rows if row["online"]),
                sum(1 for row in rows if not row["online"]),
            )
            payload = {"outcome": outcome, "servers": rows}
        elif args.command in {"job", "logs", "cancel"}:
            by_id = {target.server_id: target for target in targets}
            target = await _locate_job(transport, targets, by_id, args.run_id, args.host)
            if args.command == "job":
                detail = await transport.get_job_detail(target, args.run_id)
                if detail is None:
                    raise CommandError("job_not_found", f"job {args.run_id!r} was not found")
                payload = {
                    "server_id": target.server_id,
                    "job": detail.record.model_dump(mode="json"),
                    "explanation": detail.explanation,
                }
            elif args.command == "cancel":
                record = await transport.cancel(target, args.run_id, reason=args.reason)
                payload = {
                    "server_id": target.server_id,
                    "durable": True,
                    "job": record.model_dump(mode="json"),
                }
            else:
                payload = {
                    "server_id": target.server_id,
                    **await transport.get_logs(
                        target,
                        args.run_id,
                        tail_bytes=args.tail_bytes,
                    ),
                }
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        print_error(
            error_payload(
                error, operation=args.command, run_id=run_id, submission_started=submission_started
            ),
            as_json=args.as_json,
        )
        return 1

    if getattr(args, "as_json", False):
        print(json.dumps({"ok": outcome == "success", **payload}, ensure_ascii=False, indent=2))
    elif args.command == "logs":
        print(payload["content"], end="" if payload["content"].endswith("\n") else "\n")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if outcome == "success" else 1


def _submission_payload(submitted: SubmissionResult) -> dict[str, Any]:
    payload = {
        "run_id": submitted.run_id,
        "server_id": submitted.target.server_id,
        "submission_state": "accepted",
        "durable": True,
        "reconciled": submitted.reconciled,
        "job": submitted.record.model_dump(mode="json"),
    }
    if submitted.journal_warning:
        payload["journal_warning"] = submitted.journal_warning
    return payload


def _aggregate_outcome(success_count: int, failure_count: int) -> str:
    if failure_count == 0:
        return "success"
    if success_count == 0:
        return "failed"
    return "partial"


async def _collect_queues(transport, targets, *, full=False):  # type: ignore[no-untyped-def]
    async def collect(target):  # type: ignore[no-untyped-def]
        try:
            jobs = (
                await transport.list_jobs(target)
                if full
                else await transport.list_summaries(target, statuses=ACTIVE_JOB_STATUSES)
            )
            return {
                "server_id": target.server_id,
                "online": True,
                "jobs": [job.model_dump(mode="json") for job in jobs],
            }
        except Exception as error:
            return {
                "server_id": target.server_id,
                "online": False,
                **error_payload(error),
                "jobs": [],
            }

    return list(await asyncio.gather(*(collect(target) for target in targets)))


async def _fleet_action(transport, targets, action):  # type: ignore[no-untyped-def]
    async def execute(target):  # type: ignore[no-untyped-def]
        try:
            return {
                "server_id": target.server_id,
                "online": True,
                "result": await transport.worker_command(target, action),
            }
        except Exception as error:
            return {"server_id": target.server_id, "online": False, **error_payload(error)}

    return list(await asyncio.gather(*(execute(target) for target in targets)))


async def _locate_job(transport, targets, by_id, run_id, host):  # type: ignore[no-untyped-def]
    if host is not None:
        return by_id[host]

    async def find(target):  # type: ignore[no-untyped-def]
        try:
            found = await transport.get_job(target, run_id) is not None
            return target, found, None
        except Exception as error:
            return target, False, str(error).strip() or error.__class__.__name__

    results = await asyncio.gather(*(find(target) for target in targets))
    matches = [target for target, found, _error in results if found]
    if len(matches) != 1:
        unavailable = [
            f"{target.server_id}: {error}" for target, _found, error in results if error is not None
        ]
        if not matches and unavailable:
            checked = [
                target.server_id for target, found, error in results if not found and error is None
            ]
            details = f"; unavailable hosts: {'; '.join(unavailable)}"
            if checked:
                details += f"; not found on: {', '.join(checked)}"
            raise RuntimeError(f"job {run_id!r} could not be located{details}")
        raise RuntimeError(
            f"job {run_id!r} was {'not found' if not matches else 'found on multiple hosts'}"
        )
    return matches[0]


def _run_worker_command(args: argparse.Namespace) -> int:
    from gpuops.worker.app import (
        DEFAULT_STATE_DIRECTORY,
        render_worker_result,
        run_worker,
        start_worker,
        stop_worker,
        worker_status,
    )

    state_directory = args.state_dir or DEFAULT_STATE_DIRECTORY
    if getattr(args, "interval", 1) <= 0:
        print("gpuops: --interval must be positive", file=sys.stderr)
        return 2
    try:
        if args.worker_command == "run":
            return run_worker(
                server_id=args.server_id,
                state_directory=state_directory,
                interval_seconds=args.interval,
                once=args.once,
            )
        if args.worker_command == "start":
            result = start_worker(
                server_id=args.server_id,
                state_directory=state_directory,
                interval_seconds=args.interval,
            )
        elif args.worker_command == "stop":
            result = stop_worker(state_directory=state_directory)
        else:
            result = worker_status(state_directory=state_directory)
    except (OSError, RuntimeError, ValueError) as exc:
        print_error(error_payload(exc), as_json=args.as_json)
        return 1
    render_worker_result({"ok": True, **result}, as_json=args.as_json)
    return 0


async def _run_remote_command(args: argparse.Namespace, targets: tuple[ServerTarget, ...]) -> int:
    if not targets:
        print_error(
            error_payload(CommandError(
                "no_servers_registered",
                "No servers configured; run gpuops server add SSH_ALIAS first.",
            )),
            as_json=args.as_json,
        )
        return 1
    service = MonitorService(targets, timeout_seconds=args.timeout)
    if args.command == "status":
        from gpuops.status import collect_worker_health, format_worker_health
        from gpuops.transport import SshWorkerTransport

        results, workers = await asyncio.gather(
            service.collect_all(),
            collect_worker_health(targets, SshWorkerTransport(timeout_seconds=args.timeout)),
        )
        payload = [{**result.as_dict(), "worker": workers[result.server.id]} for result in results]
        rendered = format_status_table(results) + "\n\n" + format_worker_health(workers)
        # Preserve direct-monitor exit semantics; scheduling reads worker.ready explicitly.
        healthy = all(result.online for result in results)
    else:
        results = await service.doctor_all()
        payload = [result.as_dict() for result in results]
        rendered = format_doctor_table(results)
        healthy = all(
            result.online and bool(result.data and result.data.get("gpustat_ok"))
            for result in results
        )
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(rendered)
    return 0 if healthy else 1


def _parse_targets(values: list[str]) -> tuple[ServerTarget, ...]:
    if not values:
        return select_targets()
    targets = []
    ids = set()
    for value in values:
        server_id, separator, ssh_host = value.partition("=")
        if not separator or not server_id or not ssh_host:
            raise ValueError(f"invalid --host {value!r}; expected ID=SSH_ALIAS")
        if server_id in ids:
            raise ValueError(f"duplicate server id {server_id!r}")
        ids.add(server_id)
        targets.append(ServerTarget(id=server_id, label=server_id, ssh_host=ssh_host))
    return tuple(targets)


def _run_web(host: str, port: int, demo: bool, timeout_seconds: float) -> int:
    try:
        from gpuops.web.app import run_web
    except (ImportError, ModuleNotFoundError):
        print("gpuops: web dependencies are not installed yet", file=sys.stderr)
        return 2
    try:
        run_web(
            host=host,
            port=port,
            demo=demo,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        print(f"gpuops: {exc}", file=sys.stderr)
        return 2
    return 0


def _validate_job(path: str, as_json: bool) -> int:
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        spec = JobSpec.model_validate_json(raw)
        validate_new_submission(spec)
    except (OSError, ValidationError, ValueError) as exc:
        print_error({"valid": False, **error_payload(exc, operation="validate")}, as_json=as_json)
        return 2
    if as_json:
        print(
            json.dumps(
                {"valid": True, "job": spec.model_dump(mode="json")},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"valid gpuops/v1 job: {spec.id}")
    return 0
