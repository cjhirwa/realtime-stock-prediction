"""Running forecast-quality metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def qlike(realized: float, predicted: float) -> float:
    """QLIKE loss for a volatility forecast; robust to noise in the realized proxy."""
    ratio = (realized / predicted) ** 2
    return ratio - math.log(ratio) - 1.0


@dataclass
class ForecastMetrics:
    """Cumulative error metrics for scored return forecasts.

    `skill` compares MAE against the naive random-walk forecast (predicting a zero return):
    positive means the model beats it, zero or negative means it adds nothing.
    """

    n: int = 0
    _abs_err: float = 0.0
    _sq_err: float = 0.0
    _naive_abs_err: float = 0.0
    _dir_hits: int = 0
    _dir_n: int = 0

    def update(self, predicted: float, actual: float) -> None:
        err = actual - predicted
        self.n += 1
        self._abs_err += abs(err)
        self._sq_err += err * err
        self._naive_abs_err += abs(actual)
        if predicted != 0 and actual != 0:
            self._dir_n += 1
            self._dir_hits += (predicted > 0) == (actual > 0)

    @property
    def mae(self) -> float | None:
        return self._abs_err / self.n if self.n else None

    @property
    def rmse(self) -> float | None:
        return math.sqrt(self._sq_err / self.n) if self.n else None

    @property
    def directional_accuracy(self) -> float | None:
        return self._dir_hits / self._dir_n if self._dir_n else None

    @property
    def active_fraction(self) -> float | None:
        """Share of forecasts that made a non-zero call."""
        return self._dir_n / self.n if self.n else None

    @property
    def skill(self) -> float | None:
        if not self.n or self._naive_abs_err == 0:
            return None
        return 1.0 - self._abs_err / self._naive_abs_err

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "directional_accuracy": self.directional_accuracy,
            "active_fraction": self.active_fraction,
            "skill": self.skill,
        }

    def summary(self) -> str:
        if not self.n:
            return "no scored forecasts yet"
        return (
            f"n={self.n}  MAE={_fmt(self.mae, '.3e')}  RMSE={_fmt(self.rmse, '.3e')}  "
            f"dir_acc={_fmt(self.directional_accuracy, '.1%')}  "
            f"active={_fmt(self.active_fraction, '.0%')}  skill={_fmt(self.skill, '+.3f')}"
        )


@dataclass
class VolatilityMetrics:
    """Cumulative QLIKE for volatility forecasts against the historical-volatility baseline,
    plus empirical coverage of the forecast band."""

    target_coverage: float = 0.95
    n: int = 0
    _qlike: float = 0.0
    _baseline_qlike: float = 0.0
    _covered: int = 0
    _band_n: int = 0

    def update(
        self, realized: float, predicted: float, baseline: float, covered: bool | None
    ) -> None:
        self.n += 1
        self._qlike += qlike(realized, predicted)
        self._baseline_qlike += qlike(realized, baseline)
        if covered is not None:
            self._band_n += 1
            self._covered += covered

    @property
    def qlike(self) -> float | None:
        return self._qlike / self.n if self.n else None

    @property
    def baseline_qlike(self) -> float | None:
        return self._baseline_qlike / self.n if self.n else None

    @property
    def skill(self) -> float | None:
        if not self.n or self._baseline_qlike == 0:
            return None
        return 1.0 - self._qlike / self._baseline_qlike

    @property
    def coverage(self) -> float | None:
        return self._covered / self._band_n if self._band_n else None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "qlike": self.qlike,
            "baseline_qlike": self.baseline_qlike,
            "skill": self.skill,
            "coverage": self.coverage,
            "target_coverage": self.target_coverage,
        }

    def summary(self) -> str:
        if not self.n:
            return "no scored forecasts yet"
        return (
            f"n={self.n}  QLIKE={_fmt(self.qlike, '.3f')} "
            f"(baseline {_fmt(self.baseline_qlike, '.3f')})  skill={_fmt(self.skill, '+.3f')}  "
            f"band_coverage={_fmt(self.coverage, '.1%')} (target {self.target_coverage:.0%})"
        )


@dataclass
class SymbolMetrics:
    """All metrics tracked for one symbol."""

    returns: ForecastMetrics = field(default_factory=ForecastMetrics)
    """Return forecasts as served (after the skill gate)."""
    raw_returns: ForecastMetrics = field(default_factory=ForecastMetrics)
    """Return forecasts straight from the model, before gating."""
    volatility: VolatilityMetrics = field(default_factory=VolatilityMetrics)

    def as_dict(self) -> dict[str, dict[str, float | int | None]]:
        return {
            "returns": self.returns.as_dict(),
            "raw_returns": self.raw_returns.as_dict(),
            "volatility": self.volatility.as_dict(),
        }

    def summary(self) -> str:
        return (
            f"  returns (served): {self.returns.summary()}\n"
            f"  returns (raw):    {self.raw_returns.summary()}\n"
            f"  volatility:       {self.volatility.summary()}"
        )
