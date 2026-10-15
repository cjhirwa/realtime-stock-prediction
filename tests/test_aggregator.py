from datetime import UTC, datetime, timedelta

from stockpred.aggregator import BarAggregator
from stockpred.sources import _parse_finnhub_trades
from stockpred.types import Trade

T0 = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)


def _trade(second: int, price: float, size: float = 1.0) -> Trade:
    return Trade("AAPL", T0 + timedelta(seconds=second), price, size)


def test_emits_bar_when_next_bucket_starts():
    agg = BarAggregator(timedelta(minutes=1))
    assert agg.add(_trade(0, 10.0, 2)) is None
    assert agg.add(_trade(20, 12.0, 1)) is None
    assert agg.add(_trade(40, 9.0, 3)) is None
    bar = agg.add(_trade(61, 11.0))

    assert bar is not None
    assert bar.timestamp == T0
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (10.0, 12.0, 9.0, 9.0, 6.0)


def test_late_trade_folds_into_open_bar_and_flush_returns_it():
    agg = BarAggregator(timedelta(minutes=1))
    agg.add(_trade(61, 11.0))
    assert agg.add(_trade(30, 15.0)) is None  # late trade from previous minute
    (bar,) = agg.flush()
    assert bar.high == 15.0 and bar.close == 15.0
    assert agg.flush() == []


def test_tick_rule_signs_volume():
    agg = BarAggregator(timedelta(minutes=1))
    agg.add(_trade(0, 10.0, 5))  # no prior price: unsigned
    agg.add(_trade(10, 10.5, 2))  # uptick: buy
    agg.add(_trade(20, 10.5, 3))  # unchanged: keeps buy
    agg.add(_trade(30, 10.2, 4))  # downtick: sell
    bar = agg.add(_trade(61, 10.2, 1))
    assert bar is not None and bar.signed_volume == 2 + 3 - 4


def test_parse_finnhub_trades():
    raw = '{"type":"trade","data":[{"s":"AAPL","p":231.5,"t":1767364200000,"v":10}]}'
    (trade,) = _parse_finnhub_trades(raw)
    assert trade.symbol == "AAPL" and trade.price == 231.5 and trade.size == 10.0
    assert _parse_finnhub_trades('{"type":"ping"}') == []
