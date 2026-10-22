from __future__ import annotations

import re
from typing import Annotated, Any, TypeGuard

from pydantic import BeforeValidator

RUN_ID_MAX_LENGTH = 192
RUN_ID_PATTERN = re.compile(rf"^[A-Za-z0-9._-]{{1,{RUN_ID_MAX_LENGTH}}}$")


def is_valid_run_id(value: object) -> TypeGuard[str]:
    """Return whether value is safe as one filesystem path component."""

    return (
        isinstance(value, str)
        and value not in {".", ".."}
        and RUN_ID_PATTERN.fullmatch(value) is not None
    )


def validate_run_id(value: Any) -> str:
    if not is_valid_run_id(value):
        raise ValueError("run_id must be 1-192 ASCII characters from A-Z, a-z, 0-9, '.', '_', '-'")
    return value


RunId = Annotated[str, BeforeValidator(validate_run_id)]
