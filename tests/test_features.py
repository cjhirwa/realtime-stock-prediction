from dataclasses import replace

import numpy as np

from stockpred.features import FeatureBuilder
from stockpred.sources import SimulatedSource


def test_warmup_then_finite_vectors():
    fb = FeatureBuilder()
    bars = list(SimulatedSource(n_bars=fb.warmup + 10, seed=1).bars())

    outputs = [fb.update(b) for b in bars]

    assert all(fs is None for fs in outputs[: fb.warmup - 1])
    feature_sets = [fs for fs in outputs if fs is not None]
    assert len(feature_sets) == 11
    for fs in feature_sets:
        assert fs.returns.shape == (len(fb.return_names),)
        assert fs.volatility.shape == (len(fb.volatility_names),)
        assert np.isfinite(fs.returns).all() and np.isfinite(fs.volatility).all()
        assert fs.baseline_vol > 0


def test_order_imbalance_uses_tick_signed_volume_when_present():
    fb = FeatureBuilder()
    bars = list(SimulatedSource(n_bars=fb.warmup, seed=5).bars())
    for b in bars[:-1]:
        fb.update(b)
    last = replace(bars[-1], volume=100.0, signed_volume=-60.0)
    fs = fb.update(last)
    assert fs is not None
    assert fs.returns[fb.return_names.index("order_imbalance_1")] == -0.6


def test_order_imbalance_falls_back_to_bulk_volume_classification():
    fb = FeatureBuilder()
    fs = None
    for b in SimulatedSource(n_bars=fb.warmup + 20, seed=6).bars():
        prev, fs = fs, fb.update(b)
        if fs is not None and prev is not None:
            imbalance = fs.returns[fb.return_names.index("order_imbalance_1")]
            assert -1.0 < imbalance < 1.0
            assert np.sign(imbalance) == np.sign(fs.returns[0])  # follows the bar's return


def test_lags_and_realized_vol_match_bars():
    fb = FeatureBuilder()
    bars = list(SimulatedSource(n_bars=fb.warmup, seed=2).bars())
    fs = None
    for b in bars:
        fs = fb.update(b)
    assert fs is not None

    rets = np.diff(np.log([b.close for b in bars]))
    assert np.isclose(fs.returns[0], rets[-1])
    assert np.isclose(fs.baseline_vol, np.sqrt(np.mean(rets[-20:] ** 2)))
    assert np.isclose(fs.volatility[2], np.log(fs.baseline_vol))
