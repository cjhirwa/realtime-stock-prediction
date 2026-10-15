"""Core market-data and forecast records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed trade from a tick stream."""

    symbol: str
    timestamp: datetime
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class Bar:
    """An OHLCV bar covering one interval, stamped with the interval's start time (UTC)."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    signed_volume: float | None = None
    """Buy minus sell volume (tick rule) when built from trades; None for bar-only feeds."""


@dataclass(frozen=True, slots=True)
class Forecast:
    """A forecast of the log return from `last_close` to the close `horizon` bars later.

    `predicted_return` is what is served: zero (a random-walk forecast) while the skill gate
    is closed. `raw_predicted_return` is the model's output regardless of the gate.
    `predicted_volatility` is the forecast std of the horizon log return, and `lower`/`upper`
    bound the price at the pipeline's calibrated coverage level.
    """

    symbol: str
    timestamp: datetime
    horizon: int
    last_close: float
    predicted_return: float
    raw_predicted_return: float
    predicted_volatility: float
    predicted_price: float
    lower: float | None
    upper: float | None
    model: str

    @property
    def signal_active(self) -> bool:
        return self.predicted_return != 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["direction"] = self.direction
        data["signal_active"] = self.signal_active
        return data

    @property
    def direction(self) -> str:
        if self.predicted_return > 0:
            return "up"
        if self.predicted_return < 0:
            return "down"
        return "flat"
