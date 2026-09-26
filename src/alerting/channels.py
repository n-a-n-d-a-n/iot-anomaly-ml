"""Alert notification channels: console, file, webhook."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from src.alerting.rules_engine import Alert
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AlertChannel(ABC):
    """Base class for alert notification channels."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Deliver an alert through this channel.

        Args:
            alert: The alert to send.

        Returns:
            ``True`` if delivery succeeded.
        """


class ConsoleChannel(AlertChannel):
    """Log alerts to the console at the appropriate severity level."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("alerts")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - ALERT [%(levelname)s]: %(message)s")
            )
            self._logger.addHandler(handler)

    async def send(self, alert: Alert) -> bool:
        """Log alert to console."""
        level_map = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level = level_map.get(alert.severity, logging.WARNING)
        self._logger.log(level, alert.message)
        return True


class FileChannel(AlertChannel):
    """Append alerts to a plain-text log file.

    Args:
        filepath: Path to the output file.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    async def send(self, alert: Alert) -> bool:
        """Append alert line to file."""
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a") as fh:
            fh.write(f"{alert.timestamp.isoformat()} [{alert.severity.upper()}] {alert.message}\n")
        return True


class WebhookChannel(AlertChannel):
    """POST alerts as JSON to an HTTP endpoint.

    Args:
        url: Webhook URL.
        timeout: Request timeout in seconds.
    """

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    async def send(self, alert: Alert) -> bool:
        """Send alert payload via HTTP POST."""
        payload = {
            "timestamp": alert.timestamp.isoformat(),
            "type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "sensor_id": alert.sensor_id,
            "score": alert.anomaly_score,
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, json=payload, timeout=self.timeout)
                return response.status_code == 200
        except Exception:
            logger.exception("Webhook delivery failed")
            return False


class AlertDispatcher:
    """Fan-out alerts to all registered channels concurrently."""

    def __init__(self) -> None:
        self.channels: list[AlertChannel] = []

    def add_channel(self, channel: AlertChannel) -> None:
        """Register a notification channel.

        Args:
            channel: Channel instance to add.
        """
        self.channels.append(channel)

    async def dispatch(self, alert: Alert) -> None:
        """Send alert to every registered channel.

        Args:
            alert: The alert to broadcast.
        """
        await asyncio.gather(*(ch.send(alert) for ch in self.channels))
