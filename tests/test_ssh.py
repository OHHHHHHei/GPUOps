from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gpuops.ssh import SSHClient, SSHTimeoutError


class FakeProcess:
    def __init__(self, *, returncode=0, stdout=b"ok", stderr=b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.input = None
        self.killed = False

    async def communicate(self, data=None):
        if data is not None:
            self.input = data
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_run_python_uses_fixed_argv_and_stdin():
    process = FakeProcess(stdout=b'{"ok":true}\n')

    async def run():
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as create:
            result = await SSHClient(connect_timeout_seconds=4.2).run_python(
                "lab-gpu01", "print('hello')", timeout_seconds=2
            )
        argv = create.await_args.args
        assert argv[0] == "ssh"
        assert argv[-4:] == ("lab-gpu01", "python3", "-u", "-")
        assert "BatchMode=yes" in argv
        assert "ConnectTimeout=5" in argv
        assert process.input == b"print('hello')"
        assert result.stdout == '{"ok":true}\n'

    asyncio.run(run())


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "host name", "", "host;bad"])
def test_rejects_host_values_that_could_be_options_or_shell_syntax(host):
    with pytest.raises(ValueError):
        asyncio.run(SSHClient().run_python(host, "pass"))


def test_timeout_kills_ssh_process():
    process = FakeProcess()
    calls = 0

    async def communicate(data=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return b"", b""

    process.communicate = communicate

    async def run():
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            pytest.raises(SSHTimeoutError),
        ):
            await SSHClient().run_python("lab-gpu02", "pass", timeout_seconds=0.01)

    asyncio.run(run())
    assert process.killed


def test_run_command_quotes_remote_arguments():
    process = FakeProcess(stdout=b'{"ok":true}')

    async def run():
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as create:
            result = await SSHClient().run_command(
                "server-alias",
                ["~/.local/bin/gpuops", "remote", "job", "run with spaces"],
                stdin=b"payload",
            )
        argv = create.await_args.args
        assert argv[-1] == "~/.local/bin/gpuops remote job 'run with spaces'"
        assert process.input == b"payload"
        assert result.stdout == '{"ok":true}'

    asyncio.run(run())


def test_run_command_rejects_invalid_argv():
    with pytest.raises(ValueError):
        asyncio.run(SSHClient().run_command("server", []))
    with pytest.raises(ValueError):
        asyncio.run(SSHClient().run_command("server", ["gpuops", "bad\x00arg"]))
