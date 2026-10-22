"""On-demand GPU monitoring over SSH."""

from .collector import CollectionError, RemoteCollector
from .service import MonitorService, ServerResult, ServerTarget

__all__ = [
    "CollectionError",
    "MonitorService",
    "RemoteCollector",
    "ServerResult",
    "ServerTarget",
]
