"""Load the stdlib-only remote deployment programs shipped in the wheel."""

from __future__ import annotations

from importlib.resources import files

_BOOTSTRAP_PACKAGE = "gpuops.deployment._bootstrap"


def _source(name: str) -> str:
    return files(_BOOTSTRAP_PACKAGE).joinpath(name).read_text(encoding="utf-8")


REMOTE_PREPARE_SCRIPT = _source("prepare.py")
REMOTE_ABORT_SCRIPT = _source("abort.py")
_REMOTE_PROGRAM = _source("manage.py")
REMOTE_DEPLOY_SCRIPT = 'ACTION = "deploy"\n' + _REMOTE_PROGRAM
REMOTE_STATUS_SCRIPT = 'ACTION = "status"\n' + _REMOTE_PROGRAM
REMOTE_ROLLBACK_SCRIPT = 'ACTION = "rollback"\n' + _REMOTE_PROGRAM
REMOTE_UNINSTALL_SCRIPT = 'ACTION = "uninstall"\n' + _REMOTE_PROGRAM
