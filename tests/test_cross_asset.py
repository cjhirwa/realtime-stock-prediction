import math
from datetime import timedelta

import numpy as np

from stockpred.models import OnlineLinearForecaster
from stockpred.pipeline import ForecastPipeline
from stockpred.sources import SimulatedSource, interleave
from stockpred.types import Bar


def _lead_lag_bars(n: int, seed: int, beta: float = 0.6) -> list[Bar]:
    """'MKT' is a random walk; 'STK' follows MKT's previous-bar return with a lag."""
    rng = np.random.default_rng(seed)
    start = SimulatedSource().start
    mkt, stk, prev_mkt_ret = 100.0, 50.0, 0.0
    market, stock = [], []
    for i in range(n):
        ts = start + timedelta(minutes=i)
        mkt_ret = rng.normal(0.0, 0.001)
        stk_ret = beta * prev_mkt_ret + rng.normal(0.0, 0.0006)
        for bars, price, ret, sym in ((market, mkt, mkt_ret, "MKT"), (stock, stk, stk_ret, "STK")):
            close = price * math.exp(ret)
            hi, lo = max(price, close) * 1.0002, min(price, close) * 0.9998
            bars.append(Bar(sym, ts, price, hi, lo, close, 1e5))
        mkt, stk = mkt * math.exp(mkt_ret), stk * math.exp(stk_ret)
        prev_mkt_ret = mkt_ret
    # Shuffle the per-symbol order so interleave() must put the reference first.
    return interleave([stock, market], first=["MKT"])


def _run(reference: list[str]) -> float:
    pipeline = ForecastPipeline(OnlineLinearForecaster, reference_symbols=reference)
    for bar in _lead_lag_bars(3000, seed=21):
        pipeline.process(bar)
    skill = pipeline.metrics("STK").raw_returns.skill
    assert skill is not None
    return skill


def test_reference_symbol_captures_lead_lag():
    with_reference = _run(["MKT"])
    without_reference = _run([])
    assert with_reference > 0.1
    assert with_reference > without_reference + 0.1


def test_interleave_orders_reference_first_within_timestamp():
    bars = _lead_lag_bars(3, seed=0)
    assert [b.symbol for b in bars] == ["MKT", "STK"] * 3
