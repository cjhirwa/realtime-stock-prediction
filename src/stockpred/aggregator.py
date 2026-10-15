"""Aggregates a trade stream into fixed-interval OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from stockpred.types import Bar, Trade


@dataclass(slots=True)
class _OpenBar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    signed_volume: float

    @classmethod
    def from_trade(cls, start: datetime, trade: Trade, sign: float) -> _OpenBar:
        p = trade.price
        return cls(start, p, p, p, p, trade.size, sign * trade.size)

    def add(self, trade: Trade, sign: float) -> None:
        self.high = max(self.high, trade.price)
        self.low = min(self.low, trade.price)
        self.close = trade.price
        self.volume += trade.size
        self.signed_volume += sign * trade.size

    def to_bar(self, symbol: str) -> Bar:
        return Bar(
            symbol,
            self.start,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.signed_volume,
        )


class BarAggregator:
    """Buckets trades by interval and emits a bar once a trade lands in a later bucket.

    Each trade is signed with the tick rule (an uptick is a buy, a downtick a sell, an
    unchanged price repeats the previous sign), giving each bar a signed order-flow volume.

    A bar is only emitted when the next bucket's first trade arrives, so thinly traded
    symbols see delayed bars; call `flush()` on shutdown to retrieve the open ones.
    Late trades (older than the open bucket) are folded into the open bar.
    """

    def __init__(self, interval: timedelta) -> None:
        if interval <= timedelta(0):
            raise ValueError("interval must be positive")
        self.interval = interval
        self._open: dict[str, _OpenBar] = {}
        self._last_price: dict[str, float] = {}
        self._last_sign: dict[str, float] = {}

    def _bucket_start(self, ts: datetime) -> datetime:
        step = self.interval.total_seconds()
        epoch = ts.timestamp()
        return datetime.fromtimestamp(epoch - epoch % step, tz=UTC)

    def _tick_sign(self, trade: Trade) -> float:
        sign = self._last_sign.get(trade.symbol, 0.0)
        last = self._last_price.get(trade.symbol)
        if last is not None and trade.price != last:
            sign = 1.0 if trade.price > last else -1.0
        self._last_price[trade.symbol] = trade.price
        self._last_sign[trade.symbol] = sign
        return sign

    def add(self, trade: Trade) -> Bar | None:
        sign = self._tick_sign(trade)
        start = self._bucket_start(trade.timestamp)
        current = self._open.get(trade.symbol)
        if current is None:
            self._open[trade.symbol] = _OpenBar.from_trade(start, trade, sign)
            return None
        if start <= current.start:
            current.add(trade, sign)
            return None
        self._open[trade.symbol] = _OpenBar.from_trade(start, trade, sign)
        return current.to_bar(trade.symbol)

    def flush(self) -> list[Bar]:
        bars = [ob.to_bar(sym) for sym, ob in self._open.items()]
        self._open.clear()
        return bars
