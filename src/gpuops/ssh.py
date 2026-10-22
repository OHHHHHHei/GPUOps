"""Small async wrapper around the system OpenSSH client.

Using the system client deliberately preserves the user's existing SSH config,
keys, agents, jump hosts, and future network routing setup.
"""

from __future__ import annotations

import asyncio
import math
import re
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass

_HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:-]*$")


class SSHError(RuntimeError):
    """Base class for failures while invoking SSH."""


class SSHTimeoutError(SSHError):
    """Raised when a remote operation exceeds its deadline."""


@dataclass(frozen=True, slots=True)
class SSHResult:
    host: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


class SSHClient:
    """Run fixed Python entry points over the local OpenSSH executable."""

    def __init__(
        self,
        *,
        ssh_binary: str = "ssh",
        connect_timeout_seconds: float = 5.0,
    ) -> None:
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        self.ssh_binary = ssh_binary
        self.connect_timeout_seconds = connect_timeout_seconds

    async def run_python(
        self,
        host: str,
        script: str,
        *,
        timeout_seconds: float = 15.0,
    ) -> SSHResult:
        """Execute an in-memory Python script remotely without writing files."""

        self._validate_host(host)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        connect_timeout = max(1, math.ceil(self.connect_timeout_seconds))
        argv = (
            self.ssh_binary,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            host,
            "python3",
            "-u",
            "-",
        )

        return await self._communicate(
            host,
            argv,
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
        """Run fixed remote argv and optionally provide standard input."""

        self._validate_host(host)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        arguments = self._validate_command(command)
        connect_timeout = max(1, math.ceil(self.connect_timeout_seconds))
        argv = (
            self.ssh_binary,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            host,
            self._remote_command(arguments),
        )
        return await self._communicate(host, argv, stdin, timeout_seconds=timeout_seconds)

    @staticmethod
    async def _communicate(
        host: str,
        argv: Sequence[str],
        stdin: bytes,
        *,
        timeout_seconds: float,
    ) -> SSHResult:
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin),
                timeout=timeout_seconds,
            )
        # asyncio.TimeoutError only became an alias of TimeoutError in Python 3.11.
        except asyncio.TimeoutError as exc:  # noqa: UP041
            process.kill()
            await process.communicate()
            raise SSHTimeoutError(
                f"operation for {host!r} timed out after {timeout_seconds:g}s"
            ) from exc
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.communicate()
            raise

        return SSHResult(
            host=host,
            returncode=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            duration_seconds=time.monotonic() - started,
        )

    @staticmethod
    def _validate_command(command: Sequence[str]) -> list[str]:
        arguments = list(command)
        if not arguments or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in arguments
        ):
            raise ValueError("command must contain non-empty strings without NUL bytes")
        return arguments

    @staticmethod
    def _remote_command(arguments: list[str]) -> str:
        executable = arguments[0]
        if executable.startswith("~/") and re.fullmatch(r"~/[A-Za-z0-9_./-]+", executable):
            quoted_executable = executable
        else:
            quoted_executable = shlex.quote(executable)
        return " ".join([quoted_executable, *(shlex.quote(value) for value in arguments[1:])])

    @staticmethod
    def _validate_host(host: str) -> None:
        if not _HOST_PATTERN.fullmatch(host):
            raise ValueError(
                "SSH host must be a hostname, SSH config alias, or user@host "
                "without whitespace or options"
            )
