"""Run the local GPUOps console."""

from __future__ import annotations

import argparse

from gpuops.config import MONITOR_TIMEOUT_SECONDS
from gpuops.web.app import run_web, validate_loopback_host


def _loopback_host(value: str) -> str:
    try:
        return validate_loopback_host(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local GPUOps console")
    parser.add_argument("--host", default="127.0.0.1", type=_loopback_host)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--timeout", type=float, default=MONITOR_TIMEOUT_SECONDS)
    args = parser.parse_args()
    run_web(
        host=args.host,
        port=args.port,
        demo=args.demo,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
