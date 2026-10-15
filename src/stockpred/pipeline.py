"""Streaming forecast pipeline: bars in, forecasts out, with test-then-train learning."""

from __future__ import annotations

import math
import os
import pickle
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stockpred.features import FeatureBuilder, FeatureSet
from stockpred.metrics import SymbolMetrics, VolatilityMetrics
from stockpred.models import Forecaster, HARVolatilityForecaster, SkillGate
from stockpred.sources import BarSource
from stockpred.types import Bar, Forecast

_STATE_VERSION = 1
_CROSS_FEATURES_PER_REFERENCE = 4


@dataclass(slots=True)
class _Pending:
    due: int
    features: FeatureSet
    x_ret: np.ndarray
    """Return-model inputs: own return features plus cross-asset features."""
    raw: float
    served: float
    sigma: float
    """Model volatility of the horizon log return."""
    baseline_sigma: float
    band_q: float | None
    """Band half-width in units of `sigma`, or None before the band is calibrated."""
    close: float
    sum_sq: float = 0.0
    """Sum of squared one-bar log returns realized since the forecast."""


@dataclass
class _SymbolState:
    features: FeatureBuilder
    model: Forecaster
    vol_model: HARVolatilityForecaster
    gate: SkillGate | None
    metrics: SymbolMetrics
    pending: deque[_Pending] = field(default_factory=deque)
    norm_residuals: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    prev_close: float | None = None
    last_timestamp: datetime | None = None
    index: int = 0


class ForecastPipeline:
    """Runs one feature builder, return model, volatility model and skill gate per symbol.

    For every bar, the pipeline first scores and learns from forecasts whose horizon has
    just elapsed (so no model sees a target before predicting it), then issues a new
    forecast. The horizon is counted in bars, so gaps such as overnight closes fall inside
    a single step.

    The forecast band is `predicted_return +- q * predicted_volatility`, where q is the
    empirical `coverage` quantile of recent |error| / predicted_volatility: a split-conformal
    calibration that keeps coverage honest even when the volatility model is off.

    With `reference_symbols` (e.g. SPY), every symbol's return model also sees each
    reference's latest return, previous return and short-window mean return as of the bar's
    timestamp, plus its own return relative to the reference. Reference bars must be fed
    through the same pipeline; see `sources.interleave` for ordering replays.
    """

    def __init__(
        self,
        model_factory: Callable[[], Forecaster],
        horizon: int = 1,
        gate: bool = True,
        coverage: float = 0.95,
        min_residuals: int = 50,
        reference_symbols: Sequence[str] = (),
        feature_kwargs: dict[str, Any] | None = None,
        gate_kwargs: dict[str, Any] | None = None,
        vol_model_factory: Callable[[], HARVolatilityForecaster] = HARVolatilityForecaster,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        self.model_factory = model_factory
        self.vol_model_factory = vol_model_factory
        self.horizon = horizon
        self.gate = gate
        self.coverage = coverage
        self.min_residuals = min_residuals
        self.reference_symbols = list(dict.fromkeys(reference_symbols))
        self.feature_kwargs = feature_kwargs or {}
        self.gate_kwargs = gate_kwargs or {}
        self._states: dict[str, _SymbolState] = {}
        self._reference_returns: dict[str, deque[tuple[datetime, float]]] = {
            ref: deque(maxlen=10) for ref in self.reference_symbols
        }

    def _state(self, symbol: str) -> _SymbolState:
        state = self._states.get(symbol)
        if state is None:
            state = _SymbolState(
                features=FeatureBuilder(**self.feature_kwargs),
                model=self.model_factory(),
                vol_model=self.vol_model_factory(),
                gate=SkillGate(**self.gate_kwargs) if self.gate else None,
                metrics=SymbolMetrics(volatility=VolatilityMetrics(self.coverage)),
            )
            self._states[symbol] = state
        return state

    @property
    def symbols(self) -> list[str]:
        return list(self._states)

    def metrics(self, symbol: str) -> SymbolMetrics:
        return self._state(symbol).metrics

    def process(self, bar: Bar) -> Forecast | None:
        """Ingest one bar and return a forecast once the symbol's features are warmed up.

        Bars not newer than the symbol's last processed bar are ignored, so overlapping
        replays (e.g. a poller re-reading the day after a restart from saved state) never
        train on the same data twice.
        """
        state = self._state(bar.symbol)
        if state.last_timestamp is not None and bar.timestamp <= state.last_timestamp:
            return None
        state.last_timestamp = bar.timestamp

        if state.prev_close is not None:
            ret = math.log(bar.close / state.prev_close)
            for p in state.pending:
                p.sum_sq += ret * ret
            if bar.symbol in self._reference_returns:
                self._reference_returns[bar.symbol].append((bar.timestamp, ret))
        state.prev_close = bar.close

        while state.pending and state.pending[0].due <= state.index:
            self._resolve(state, state.pending.popleft(), bar)

        forecast = None
        features = state.features.update(bar)
        if features is not None:
            forecast = self._forecast(state, bar, features)
        state.index += 1
        return forecast

    async def run(self, source: BarSource) -> AsyncIterator[Forecast]:
        async for bar in source.stream():
            forecast = self.process(bar)
            if forecast is not None:
                yield forecast

    def _cross_features(self, bar: Bar, own_return: float) -> list[float]:
        out: list[float] = []
        for ref in self.reference_symbols:
            history = [r for ts, r in self._reference_returns[ref] if ts <= bar.timestamp]
            if ref == bar.symbol or not history:
                out.extend([0.0] * _CROSS_FEATURES_PER_REFERENCE)
                continue
            latest = history[-1]
            previous = history[-2] if len(history) > 1 else 0.0
            out.extend([latest, previous, float(np.mean(history[-5:])), own_return - latest])
        return out

    def _resolve(self, state: _SymbolState, p: _Pending, bar: Bar) -> None:
        h = self.horizon
        actual = math.log(bar.close / p.close)
        floor = 0.1 * p.features.baseline_vol
        rv_bar = max(math.sqrt(p.sum_sq / h), floor, 1e-10)
        norm_err = abs(actual - p.served) / p.sigma

        m = state.metrics
        m.returns.update(p.served, actual)
        m.raw_returns.update(p.raw, actual)
        m.volatility.update(
            realized=rv_bar * math.sqrt(h),
            predicted=p.sigma,
            baseline=p.baseline_sigma,
            covered=None if p.band_q is None else norm_err <= p.band_q,
        )
        state.norm_residuals.append(norm_err)
        if state.gate is not None:
            state.gate.update(p.raw, actual)
        state.model.learn(p.x_ret, actual)
        state.vol_model.learn(p.features.volatility, rv_bar)

    def _forecast(self, state: _SymbolState, bar: Bar, features: FeatureSet) -> Forecast:
        h_scale = math.sqrt(self.horizon)
        baseline_sigma = max(features.baseline_vol, 1e-10) * h_scale
        sigma = (
            state.vol_model.predict(features.volatility) * h_scale
            if state.vol_model.ready
            else baseline_sigma
        )
        x_ret = features.returns
        if self.reference_symbols:
            cross = self._cross_features(bar, float(features.returns[0]))
            x_ret = np.concatenate([x_ret, cross])
        raw = state.model.predict(x_ret)
        served = raw if state.gate is None else state.gate.apply(raw)

        band_q = None
        lower = upper = None
        if len(state.norm_residuals) >= self.min_residuals:
            band_q = float(np.quantile(state.norm_residuals, self.coverage))
            lower = bar.close * math.exp(served - band_q * sigma)
            upper = bar.close * math.exp(served + band_q * sigma)

        state.pending.append(
            _Pending(
                due=state.index + self.horizon,
                features=features,
                x_ret=x_ret,
                raw=raw,
                served=served,
                sigma=sigma,
                baseline_sigma=baseline_sigma,
                band_q=band_q,
                close=bar.close,
            )
        )
        return Forecast(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            horizon=self.horizon,
            last_close=bar.close,
            predicted_return=served,
            raw_predicted_return=raw,
            predicted_volatility=sigma,
            predicted_price=bar.close * math.exp(served),
            lower=lower,
            upper=upper,
            model=state.model.name,
        )

    # -- persistence ---------------------------------------------------------------------

    def _config(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "gate": self.gate,
            "coverage": self.coverage,
            "reference_symbols": self.reference_symbols,
            "feature_kwargs": self.feature_kwargs,
            "model": self.model_factory().name,
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write learned state (features, models, gates, metrics) atomically to `path`.

        Unresolved forecasts are not saved: after a restart, their horizons would span the
        downtime and score against the wrong bars.
        """
        payload = {
            "version": _STATE_VERSION,
            "config": self._config(),
            "states": {sym: replace(st, pending=deque()) for sym, st in self._states.items()},
            "reference_returns": self._reference_returns,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path: str | os.PathLike[str]) -> None:
        """Restore state written by `save`.

        Only load files this program wrote: the format is pickle, which can execute code.
        Raises ValueError if the file was saved with a different configuration.
        """
        with Path(path).open("rb") as f:
            payload = pickle.load(f)  # noqa: S301 - trusted local state file
        if payload.get("version") != _STATE_VERSION:
            raise ValueError(f"unsupported state version {payload.get('version')!r}")
        saved, current = payload["config"], self._config()
        mismatched = sorted(k for k in current if saved.get(k) != current[k])
        if mismatched:
            details = ", ".join(
                f"{k}: saved={saved.get(k)!r} now={current[k]!r}" for k in mismatched
            )
            raise ValueError(f"state file config does not match pipeline ({details})")
        self._states = payload["states"]
        self._reference_returns = payload["reference_returns"]
