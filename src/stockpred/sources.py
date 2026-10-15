"""Market-data sources, each exposing an async stream of bars."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from stockpred.aggregator import BarAggregator
from stockpred.types import Bar, Trade

log = logging.getLogger(__name__)

FINNHUB_WS_URL = "wss://ws.finnhub.io"


class DataUnavailableError(RuntimeError):
    """Raised when a provider returns no data for a request."""


class BarSource(ABC):
    @abstractmethod
    def stream(self) -> AsyncIterator[Bar]:
        """Yield bars as they complete."""


class ReplaySource(BarSource):
    """Replays a fixed sequence of bars, optionally paced by `delay` seconds per bar."""

    def __init__(self, bars: Iterable[Bar], delay: float = 0.0) -> None:
        self._bars = bars
        self.delay = delay

    async def stream(self) -> AsyncIterator[Bar]:
        for bar in self._bars:
            yield bar
            if self.delay:
                await asyncio.sleep(self.delay)


class MergedSource(BarSource):
    """Merges several sources into one stream, yielding bars as each source produces them.

    An error in any source stops the merged stream and is re-raised to the consumer.
    """

    def __init__(self, sources: Sequence[BarSource], queue_size: int = 1000) -> None:
        self.sources = list(sources)
        self.queue_size = queue_size

    async def stream(self) -> AsyncIterator[Bar]:
        queue: asyncio.Queue[Bar | BaseException | None] = asyncio.Queue(self.queue_size)

        async def pump(source: BarSource) -> None:
            try:
                async for bar in source.stream():
                    await queue.put(bar)
            except Exception as exc:
                await queue.put(exc)
            else:
                await queue.put(None)

        tasks = [asyncio.create_task(pump(s)) for s in self.sources]
        remaining = len(tasks)
        try:
            while remaining:
                item = await queue.get()
                if item is None:
                    remaining -= 1
                elif isinstance(item, BaseException):
                    raise item
                else:
                    yield item
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def interleave(bar_lists: Iterable[Sequence[Bar]], first: Iterable[str] = ()) -> list[Bar]:
    """Merge per-symbol histories into one chronological replay.

    Bars sharing a timestamp put symbols in `first` (e.g. reference symbols) ahead of the
    rest, so cross-asset features can use the reference's bar for the same interval.
    """
    priority = set(first)
    bars = [bar for bars in bar_lists for bar in bars]
    return sorted(bars, key=lambda b: (b.timestamp, b.symbol not in priority, b.symbol))


class SimulatedSource(ReplaySource):
    """Synthetic prices whose returns follow AR(1) with drift: r_t = mu + phi * r_{t-1} + eps.

    The autocorrelation `phi` gives models a real, known signal to recover, which makes
    this source useful for development and for testing that learning actually works.
    """

    def __init__(
        self,
        symbol: str = "SIM",
        n_bars: int | None = None,
        interval: timedelta = timedelta(minutes=1),
        start_price: float = 100.0,
        mu: float = 0.0,
        phi: float = 0.2,
        sigma: float = 0.001,
        seed: int | None = None,
        delay: float = 0.0,
        start: datetime | None = None,
    ) -> None:
        super().__init__(self.bars(), delay)
        self.symbol = symbol
        self.n_bars = n_bars
        self.interval = interval
        self.start_price = start_price
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.seed = seed
        self.start = start or datetime(2026, 1, 2, 14, 30, tzinfo=UTC)

    def bars(self) -> Iterator[Bar]:
        rng = np.random.default_rng(self.seed)
        price, prev_ret, ts = self.start_price, 0.0, self.start
        i = 0
        while self.n_bars is None or i < self.n_bars:
            ret = self.mu + self.phi * prev_ret + rng.normal(0.0, self.sigma)
            close = price * math.exp(ret)
            wick = abs(rng.normal(0.0, self.sigma / 2))
            yield Bar(
                symbol=self.symbol,
                timestamp=ts,
                open=price,
                high=max(price, close) * (1 + wick),
                low=min(price, close) * (1 - wick),
                close=close,
                volume=float(rng.lognormal(10.0, 0.5)),
            )
            price, prev_ret, ts = close, ret, ts + self.interval
            i += 1


def load_history(symbol: str, period: str = "5d", interval: str = "1m") -> list[Bar]:
    """Download historical bars from Yahoo Finance (no API key required)."""
    import yfinance as yf

    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise DataUnavailableError(f"no data for {symbol} (period={period}, interval={interval})")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return [
        Bar(
            symbol=symbol,
            timestamp=ts.to_pydatetime(),
            open=float(row.Open),
            high=float(row.High),
            low=float(row.Low),
            close=float(row.Close),
            volume=float(row.Volume),
        )
        for ts, row in zip(index, df.itertuples(), strict=True)
    ]


class YFinancePollingSource(BarSource):
    """Polls Yahoo Finance and emits each completed bar once.

    The first poll replays the whole `lookback` window, which warms up features and
    models. The newest bar is held back because it is still forming.
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        poll_seconds: float = 30.0,
        lookback: str = "1d",
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.poll_seconds = poll_seconds
        self.lookback = lookback

    async def stream(self) -> AsyncIterator[Bar]:
        last_ts: datetime | None = None
        while True:
            try:
                bars = await asyncio.to_thread(
                    load_history, self.symbol, self.lookback, self.interval
                )
            except Exception as exc:  # network/provider errors: keep polling
                log.warning("yfinance poll failed for %s: %s", self.symbol, exc)
                bars = []
            for bar in bars[:-1]:
                if last_ts is None or bar.timestamp > last_ts:
                    last_ts = bar.timestamp
                    yield bar
            await asyncio.sleep(self.poll_seconds)


class FinnhubSource(BarSource):
    """Streams trades over Finnhub's WebSocket API and aggregates them into bars.

    Requires a (free) API key from https://finnhub.io. US equities only trade during
    market hours; crypto symbols such as "BINANCE:BTCUSDT" stream around the clock.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        api_key: str,
        interval: timedelta = timedelta(minutes=1),
        max_backoff: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("Finnhub API key is required (set FINNHUB_API_KEY)")
        self.symbols = list(symbols)
        self.api_key = api_key
        self.interval = interval
        self.max_backoff = max_backoff

    async def stream(self) -> AsyncIterator[Bar]:
        import websockets

        aggregator = BarAggregator(self.interval)
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(f"{FINNHUB_WS_URL}?token={self.api_key}") as ws:
                    for symbol in self.symbols:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                    log.info("subscribed to %s", ", ".join(self.symbols))
                    backoff = 1.0
                    async for raw in ws:
                        for trade in _parse_finnhub_trades(raw):
                            bar = aggregator.add(trade)
                            if bar is not None:
                                yield bar
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                log.warning("Finnhub connection lost (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)


def _parse_finnhub_trades(raw: str | bytes) -> list[Trade]:
    msg = json.loads(raw)
    if msg.get("type") != "trade":
        return []
    return [
        Trade(
            symbol=t["s"],
            timestamp=datetime.fromtimestamp(t["t"] / 1000, tz=UTC),
            price=float(t["p"]),
            size=float(t.get("v", 0.0)),
        )
        for t in msg.get("data", [])
    ]
