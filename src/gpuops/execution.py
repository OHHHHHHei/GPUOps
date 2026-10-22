"""Execute on this GPUOps installation directly, or use the configured SSH route."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from gpuops.config import local_target
from gpuops.ssh import SSHClient, SSHResult


class HostClient(SSHClient):
    """Share argv/stdin execution and process cleanup with the SSH client."""

    def __init__(
        self, *, ssh_binary: str = "ssh", connect_timeout_seconds: float = 5.0
    ) -> None:
        super().__init__(
            ssh_binary=ssh_binary, connect_timeout_seconds=connect_timeout_seconds
        )
        self.local = local_target()

    def _is_local(self, host: str) -> bool:
        return self.local is not None and host == self.local.ssh_host

    async def run_python(
        self, host: str, script: str, *, timeout_seconds: float = 15.0
    ) -> SSHResult:
        if not self._is_local(host):
            return await super().run_python(host, script, timeout_seconds=timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return await self._communicate(
            host,
            (sys.executable, "-I", "-u", "-"),
            script.encode("utf-8"),
            timeout_seconds=timeout_seconds,
        )

    async def run_command(
        self,
        host: str,
        command: Sequence[str],
        *,
        stdin: bytes = b"",
        timeout_seconds: float = 15.0,
    ) -> SSHResult:
        if not self._is_local(host):
            return await super().run_command(
                host, command, stdin=stdin, timeout_seconds=timeout_seconds
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        arguments = self._validate_command(command)
        # The installed entry point pins its Python environment and works from any cwd.
        if arguments[0].startswith("~/"):
            arguments[0] = str(Path.home() / arguments[0][2:])
        return await self._communicate(host, arguments, stdin, timeout_seconds=timeout_seconds)
