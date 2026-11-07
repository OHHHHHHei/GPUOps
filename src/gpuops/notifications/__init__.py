"""Host-local SMTP notifications for terminal jobs."""

from gpuops.notifications.config import (
    NotificationConfig,
    NotificationConfigError,
    SMTPSettings,
    config_path,
    configure,
    disable,
    load_config,
)
from gpuops.notifications.mail import MailDeliveryError, send_test
from gpuops.notifications.service import JobNotifier, notification_status

__all__ = [
    "JobNotifier",
    "MailDeliveryError",
    "NotificationConfig",
    "NotificationConfigError",
    "SMTPSettings",
    "config_path",
    "configure",
    "disable",
    "load_config",
    "notification_status",
    "send_test",
]
