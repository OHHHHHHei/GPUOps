"""Shared ordering and opaque cursor for the multi-server job history."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone

HistoryKey = tuple[datetime, str, str]


def encode_cursor(key: HistoryKey) -> str:
    finished_at, run_id, server_id = key
    payload = json.dumps(
        [finished_at.astimezone(timezone.utc).isoformat(), run_id, server_id],
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> HistoryKey | None:
    if cursor is None:
        return None
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 1024:
            raise ValueError
        values = json.loads(
            base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        )
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError
        finished_at = datetime.fromisoformat(values[0])
        if finished_at.tzinfo is None:
            raise ValueError
        return finished_at.astimezone(timezone.utc), values[1], values[2]
    except (ValueError, TypeError, OverflowError, binascii.Error) as error:
        raise ValueError("invalid history cursor") from error
