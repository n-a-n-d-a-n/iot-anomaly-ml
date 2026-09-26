"""Stream simulator that replays pre-generated sensor data as async events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SensorReading:
    """Single sensor reading emitted by the stream.

    Attributes:
        timestamp: Reading timestamp.
        sensor_id: Unique sensor identifier.
        sensor_type: Category of sensor (temperature, vibration, etc.).
        value: Measured sensor value.
    """

    timestamp: datetime
    sensor_id: str
    sensor_type: str
    value: float


class StreamSimulator:
    """Replay a sensor DataFrame as an asynchronous stream.

    Args:
        data: Long-format sensor DataFrame with ``timestamp``, ``sensor_id``,
            ``sensor_type``, and ``value`` columns.
        speed_multiplier: Factor to accelerate replay (e.g. 1000 = 1000x real-time).
    """

    def __init__(self, data: pd.DataFrame, speed_multiplier: float = 1.0) -> None:
        self.data = data.sort_values("timestamp")
        self.speed_multiplier = speed_multiplier
        self._running: bool = False

    async def stream(self) -> AsyncIterator[SensorReading]:
        """Yield sensor readings simulating real-time arrival.

        Yields:
            :class:`SensorReading` instances in timestamp order.
        """
        self._running = True
        prev_time: pd.Timestamp | None = None

        for _, row in self.data.iterrows():
            if not self._running:
                break

            current_time = pd.to_datetime(row["timestamp"])
            if prev_time is not None:
                delay = (current_time - prev_time).total_seconds() / self.speed_multiplier
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))

            prev_time = current_time
            yield SensorReading(
                timestamp=current_time.to_pydatetime(),
                sensor_id=row["sensor_id"],
                sensor_type=row["sensor_type"],
                value=float(row["value"]),
            )

    def stop(self) -> None:
        """Signal the stream to stop yielding."""
        self._running = False
