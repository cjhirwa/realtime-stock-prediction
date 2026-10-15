import numpy as np
import pytest

from stockpred.models import OnlineLinearForecaster
from stockpred.pipeline import ForecastPipeline
from stockpred.sources import SimulatedSource


def _trained(n_bars: int = 600) -> tuple[ForecastPipeline, list]:
    bars = list(SimulatedSource(n_bars=n_bars + 50, phi=0.3, seed=9).bars())
    pipeline = ForecastPipeline(OnlineLinearForecaster)
    for bar in bars[:n_bars]:
        pipeline.process(bar)
    return pipeline, bars[n_bars:]


def test_save_and_load_round_trip(tmp_path):
    original, _ = _trained()
    path = tmp_path / "state" / "pipeline.pkl"
    original.save(path)
    assert path.exists() and not path.with_name("pipeline.pkl.tmp").exists()

    restored = ForecastPipeline(OnlineLinearForecaster)
    restored.load(path)

    assert restored.symbols == ["SIM"]
    a, b = original._state("SIM"), restored._state("SIM")
    np.testing.assert_array_equal(a.model.weights, b.model.weights)
    assert a.metrics.returns.n == b.metrics.returns.n
    assert a.features.warmup == b.features.warmup
    assert len(b.pending) == 0  # unresolved forecasts are dropped on save


def test_restored_pipeline_forecasts_immediately(tmp_path):
    original, rest = _trained()
    original.save(tmp_path / "s.pkl")
    restored = ForecastPipeline(OnlineLinearForecaster)
    restored.load(tmp_path / "s.pkl")

    forecast = restored.process(rest[0])  # no warmup needed
    expected = original.process(rest[0])
    assert forecast is not None and expected is not None
    # Same features and volatility model; the return model may differ only by the
    # unresolved forecasts the original still learns from.
    assert forecast.predicted_volatility > 0
    assert forecast.last_close == expected.last_close


def test_restored_pipeline_skips_already_processed_bars(tmp_path):
    bars = list(SimulatedSource(n_bars=300, seed=10).bars())
    original = ForecastPipeline(OnlineLinearForecaster)
    for bar in bars[:200]:
        original.process(bar)
    original.save(tmp_path / "s.pkl")

    restored = ForecastPipeline(OnlineLinearForecaster)
    restored.load(tmp_path / "s.pkl")
    n_before = restored.metrics("SIM").returns.n
    # Replay overlapping history, as a poller does on its first poll after a restart.
    assert all(restored.process(bar) is None for bar in bars[100:200])
    assert restored.metrics("SIM").returns.n == n_before
    assert restored.process(bars[200]) is not None


def test_load_rejects_mismatched_config(tmp_path):
    original, _ = _trained(100)
    original.save(tmp_path / "s.pkl")
    with pytest.raises(ValueError, match="horizon"):
        ForecastPipeline(OnlineLinearForecaster, horizon=5).load(tmp_path / "s.pkl")
