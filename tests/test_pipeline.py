import asyncio
import math
from datetime import timedelta

import numpy as np
import pytest

from stockpred.metrics import ForecastMetrics, VolatilityMetrics, qlike
from stockpred.models import MODELS, NaiveForecaster, OnlineLinearForecaster, SkillGate
from stockpred.pipeline import ForecastPipeline
from stockpred.sources import SimulatedSource
from stockpred.types import Bar


def test_return_metrics_basic():
    m = ForecastMetrics()
    m.update(0.01, 0.02)
    m.update(-0.01, 0.01)
    m.update(0.0, 0.01)
    assert m.n == 3
    assert m.mae == pytest.approx(0.04 / 3)
    assert m.directional_accuracy == 0.5
    assert m.active_fraction == pytest.approx(2 / 3)
    assert m.skill == pytest.approx(1 - 0.04 / 0.04)


def test_volatility_metrics_basic():
    assert qlike(0.01, 0.01) == pytest.approx(0.0)
    assert qlike(0.01, 0.005) > qlike(0.01, 0.02)  # QLIKE punishes under-prediction harder
    m = VolatilityMetrics()
    m.update(0.01, 0.01, 0.02, covered=True)
    m.update(0.01, 0.01, 0.02, covered=None)
    assert m.skill == pytest.approx(1.0)
    assert m.coverage == 1.0


def test_skill_gate_opens_only_with_positive_recent_skill():
    gate = SkillGate(window=50, min_samples=20)
    for _ in range(19):
        gate.update(0.01, 0.01)
    assert not gate.is_open  # too few samples
    gate.update(0.01, 0.01)
    assert gate.is_open and gate.apply(0.003) == 0.003
    for _ in range(50):
        gate.update(0.01, -0.01)
    assert not gate.is_open and gate.apply(0.003) == 0.0


@pytest.mark.parametrize("horizon", [1, 3])
def test_forecasts_scored_after_horizon(horizon):
    pipeline = ForecastPipeline(NaiveForecaster, horizon=horizon)
    bars = list(SimulatedSource(n_bars=200, seed=3).bars())
    forecasts = [f for b in bars if (f := pipeline.process(b)) is not None]

    m = pipeline.metrics("SIM")
    assert m.returns.n == m.volatility.n == len(forecasts) - horizon
    f = forecasts[-1]
    assert f.predicted_return == 0.0 and f.predicted_price == pytest.approx(f.last_close)
    assert f.predicted_volatility > 0
    assert f.lower is not None and f.lower < f.last_close < f.upper


def test_async_run_yields_forecasts():
    async def collect():
        pipeline = ForecastPipeline(MODELS["ewma"])
        return [f async for f in pipeline.run(SimulatedSource(n_bars=50, seed=4))]

    forecasts = asyncio.run(collect())
    assert len(forecasts) == 50 - 21 + 1
    assert all(math.isfinite(f.predicted_price) for f in forecasts)


def test_online_model_learns_autocorrelated_signal():
    """On AR(1) returns with phi=0.3 the online model should beat the random walk."""
    pipeline = ForecastPipeline(OnlineLinearForecaster)
    for bar in SimulatedSource(n_bars=4000, phi=0.3, seed=7).bars():
        pipeline.process(bar)

    m = pipeline.metrics("SIM").returns
    assert m.skill is not None and m.skill > 0.0
    assert m.directional_accuracy is not None and m.directional_accuracy > 0.55
    assert m.active_fraction is not None and m.active_fraction > 0.9


def test_gate_stays_closed_without_signal():
    """On a pure random walk the gate should keep served forecasts at (near) zero."""
    pipeline = ForecastPipeline(OnlineLinearForecaster)
    for bar in SimulatedSource(n_bars=3000, phi=0.0, seed=8).bars():
        pipeline.process(bar)

    m = pipeline.metrics("SIM")
    assert m.returns.active_fraction is not None and m.returns.active_fraction < 0.2
    assert m.returns.skill is not None and m.returns.skill > m.raw_returns.skill


def _seasonal_vol_bars(n: int, seed: int) -> list[Bar]:
    """Random walk whose volatility follows a daily cycle, like intraday U-shaped volatility."""
    rng = np.random.default_rng(seed)
    bars, price = [], 100.0
    start = SimulatedSource().start
    for i in range(n):
        ts = start + timedelta(minutes=i)
        angle = 2 * math.pi * (ts.hour * 60 + ts.minute) / 1440
        sigma = 0.001 * (1 + 2 * math.cos(angle) ** 2)
        close = price * math.exp(rng.normal(0.0, sigma))
        wick = abs(rng.normal(0.0, sigma / 2))
        high, low = max(price, close) * (1 + wick), min(price, close) * (1 - wick)
        bars.append(Bar("SIM", ts, price, high, low, close, float(rng.lognormal(10, 0.5))))
        price = close
    return bars


def test_volatility_model_beats_historical_baseline_and_band_is_calibrated():
    pipeline = ForecastPipeline(NaiveForecaster, coverage=0.9)
    for bar in _seasonal_vol_bars(6000, seed=11):
        pipeline.process(bar)

    vm = pipeline.metrics("SIM").volatility
    assert vm.skill is not None and vm.skill > 0.0
    assert vm.coverage is not None and abs(vm.coverage - 0.9) < 0.03
