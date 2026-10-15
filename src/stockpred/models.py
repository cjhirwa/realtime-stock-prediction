"""Online forecasters that predict the log return over the forecast horizon."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np


class Forecaster(Protocol):
    name: str

    def predict(self, x: np.ndarray) -> float:
        """Predict the horizon log return for feature vector `x`."""
        ...

    def learn(self, x: np.ndarray, y: float) -> None:
        """Update on a realized (features, horizon log return) pair."""
        ...


class RunningStats:
    """Welford's online mean and variance."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0


class NaiveForecaster:
    """Random-walk baseline: the best guess for the next price is the current price."""

    name = "naive"

    def predict(self, x: np.ndarray) -> float:
        return 0.0

    def learn(self, x: np.ndarray, y: float) -> None:
        pass


class EWMAForecaster:
    """Drift baseline: predicts an exponentially weighted mean of recent realized returns."""

    name = "ewma"

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._mean = 0.0

    def predict(self, x: np.ndarray) -> float:
        return self._mean

    def learn(self, x: np.ndarray, y: float) -> None:
        self._mean += self.alpha * (y - self._mean)


class VectorRunningStats:
    """Welford's online mean and variance, per component of a vector."""

    def __init__(self) -> None:
        self.n = 0
        self.mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        if self.mean is None or self._m2 is None:
            self.mean = np.zeros_like(x, dtype=float)
            self._m2 = np.zeros_like(x, dtype=float)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self._m2 is None or self.n < 2:
            return np.zeros_like(x, dtype=float)
        std = np.sqrt(self._m2 / (self.n - 1))
        return np.divide(x - self.mean, std, out=np.zeros_like(x, dtype=float), where=std > 0)


class OnlineLinearForecaster:
    """Online linear regression on the engineered features, i.e. an adaptive AR(p) model
    with exogenous terms.

    Features are standardized with running statistics and the target is scaled by its
    running std, so SGD sees unit-scale inputs regardless of the symbol's volatility. A
    constant learning rate keeps the model adapting as market regimes drift; Huber loss
    limits the pull of outlier returns, and L2 regularization shrinks noisy weights.
    """

    name = "sgd"

    def __init__(
        self,
        learning_rate: float = 0.005,
        l2: float = 1e-4,
        huber_delta: float = 1.35,
        min_samples: int = 30,
        max_abs_z: float = 5.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.huber_delta = huber_delta
        self.min_samples = min_samples
        self.max_abs_z = max_abs_z
        self.n_seen = 0
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self._features = VectorRunningStats()
        self._target = RunningStats()

    def _raw_predict(self, z: np.ndarray) -> float:
        return 0.0 if self.weights is None else float(z @ self.weights) + self.bias

    def predict(self, x: np.ndarray) -> float:
        if self.n_seen < self.min_samples:
            return self._target.mean
        yz = self._raw_predict(self._features.standardize(x))
        yz = float(np.clip(yz, -self.max_abs_z, self.max_abs_z))
        return self._target.mean + yz * self._target.std

    def learn(self, x: np.ndarray, y: float) -> None:
        self._target.update(y)
        self._features.update(x)
        scale = self._target.std
        if scale == 0:
            return
        if self.weights is None:
            self.weights = np.zeros_like(x, dtype=float)
        z = self._features.standardize(x)
        target = (y - self._target.mean) / scale
        # Huber gradient: linear error inside delta, clipped beyond it.
        grad = float(np.clip(self._raw_predict(z) - target, -self.huber_delta, self.huber_delta))
        self.weights -= self.learning_rate * (grad * z + self.l2 * self.weights)
        self.bias -= self.learning_rate * grad
        self.n_seen += 1


class HARVolatilityForecaster:
    """Forecasts per-bar realized volatility over the horizon.

    An online linear regression on log volatility (HAR: short, medium and long realized
    volatility, plus range, volume and time-of-day terms). exp(E[log rv]) underestimates
    the level, so the output is rescaled by a running estimate of E[rv^2 / forecast^2],
    the scaling that minimizes QLIKE loss.
    """

    name = "har"

    def __init__(
        self,
        calibration_alpha: float = 0.01,
        learning_rate: float = 0.003,
        min_samples: int = 30,
    ) -> None:
        self.calibration_alpha = calibration_alpha
        self._reg = OnlineLinearForecaster(learning_rate=learning_rate, min_samples=min_samples)
        self._calibration = 1.0

    @property
    def ready(self) -> bool:
        return self._reg.n_seen >= self._reg.min_samples

    def predict(self, x: np.ndarray) -> float:
        return math.exp(self._reg.predict(x)) * math.sqrt(self._calibration)

    def learn(self, x: np.ndarray, rv: float) -> None:
        if self.ready:
            ratio = min((rv / math.exp(self._reg.predict(x))) ** 2, 100.0)
            self._calibration += self.calibration_alpha * (ratio - self._calibration)
        self._reg.learn(x, math.log(rv))


class SkillGate:
    """Serves a return forecast only while the model's recent out-of-sample skill is positive.

    Otherwise the served forecast is zero, the random-walk forecast, while the model keeps
    learning in the background. This stops a model without an edge from adding error.
    """

    def __init__(self, window: int = 300, min_samples: int = 100, threshold: float = 0.0) -> None:
        self.min_samples = min_samples
        self.threshold = threshold
        self._abs_actual: deque[float] = deque(maxlen=window)
        self._abs_error: deque[float] = deque(maxlen=window)

    def update(self, raw_predicted: float, actual: float) -> None:
        self._abs_actual.append(abs(actual))
        self._abs_error.append(abs(actual - raw_predicted))

    @property
    def recent_skill(self) -> float | None:
        total = sum(self._abs_actual)
        if len(self._abs_actual) < self.min_samples or total == 0:
            return None
        return 1.0 - sum(self._abs_error) / total

    @property
    def is_open(self) -> bool:
        skill = self.recent_skill
        return skill is not None and skill > self.threshold

    def apply(self, raw_predicted: float) -> float:
        return raw_predicted if self.is_open else 0.0


MODELS: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "ewma": EWMAForecaster,
    "sgd": OnlineLinearForecaster,
}
