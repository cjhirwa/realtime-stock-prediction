"""Incremental feature extraction from a bar stream."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from stockpred.types import Bar

_PARKINSON = 1.0 / (2.0 * math.sqrt(math.log(2.0)))


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Model inputs derived from the bars up to and including the latest one."""

    returns: np.ndarray
    """Inputs to the return model."""
    volatility: np.ndarray
    """Inputs to the volatility model."""
    baseline_vol: float
    """Per-bar realized volatility over the long window: the historical-volatility baseline."""


class FeatureBuilder:
    """Maintains a rolling window for one symbol and turns each new bar into features.

    All features are scale-free, so they transfer across symbols and price levels.

    Return features: the last `n_lags` log returns; mean log return over the short and
    long windows; std of log returns over the long window; RSI rescaled to [-1, 1]; close
    relative to its EMA; bar range (high - low) / close; z-score of log volume; order-flow
    imbalance (signed / total volume) over the last bar and the short window. Imbalance
    uses the bar's tick-rule `signed_volume` when present, otherwise bulk volume
    classification: buy share = Phi(return / volatility).

    Volatility features (HAR-style, in logs): realized volatility over the last bar, the
    short window and the long window; Parkinson high-low volatility of the last bar and
    averaged over the short window; volume z-score; time of day as two Fourier harmonics,
    which lets a linear model fit the intraday U-shape of volatility.
    """

    def __init__(
        self,
        n_lags: int = 5,
        short_window: int = 5,
        long_window: int = 20,
        rsi_period: int = 14,
    ) -> None:
        self.n_lags = n_lags
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.warmup = max(n_lags, short_window, long_window, rsi_period) + 1
        self._closes: deque[float] = deque(maxlen=self.warmup)
        self._log_ranges: deque[float] = deque(maxlen=short_window)
        self._log_volumes: deque[float] = deque(maxlen=long_window)
        self._volumes: deque[float] = deque(maxlen=short_window)
        self._signed_volumes: deque[float] = deque(maxlen=short_window)
        self._ema: float | None = None
        self._ema_alpha = 2.0 / (long_window + 1)
        self.return_names = [f"ret_lag_{i}" for i in range(1, n_lags + 1)] + [
            f"ret_mean_{short_window}",
            f"ret_mean_{long_window}",
            f"ret_std_{long_window}",
            f"rsi_{rsi_period}",
            f"close_ema_{long_window}",
            "bar_range",
            f"volume_z_{long_window}",
            "order_imbalance_1",
            f"order_imbalance_{short_window}",
        ]
        self.volatility_names = [
            "log_rv_1",
            f"log_rv_{short_window}",
            f"log_rv_{long_window}",
            "log_parkinson_1",
            f"log_parkinson_{short_window}",
            f"volume_z_{long_window}",
            "tod_sin_1",
            "tod_cos_1",
            "tod_sin_2",
            "tod_cos_2",
        ]

    def update(self, bar: Bar) -> FeatureSet | None:
        """Ingest a bar; return its features, or None while still warming up."""
        self._closes.append(bar.close)
        self._log_ranges.append(math.log(bar.high / bar.low) if bar.low > 0 else 0.0)
        self._log_volumes.append(math.log1p(max(bar.volume, 0.0)))
        self._ema = (
            bar.close
            if self._ema is None
            else self._ema + self._ema_alpha * (bar.close - self._ema)
        )
        if len(self._closes) < self.warmup:
            return None

        closes = np.asarray(self._closes)
        rets = np.diff(np.log(closes))
        long_rets = rets[-self.long_window :]
        short_rets = rets[-self.short_window :]

        diffs = np.diff(closes)[-self.rsi_period :]
        gain = diffs[diffs > 0].sum()
        loss = -diffs[diffs < 0].sum()
        rsi = 0.0 if gain + loss == 0 else (gain - loss) / (gain + loss)  # == RSI/50 - 1

        vols = np.asarray(self._log_volumes)
        vol_std = vols.std()
        volume_z = 0.0 if vol_std == 0 else (vols[-1] - vols.mean()) / vol_std

        volume = max(bar.volume, 0.0)
        if bar.signed_volume is not None:
            signed = bar.signed_volume
        else:
            ret_std = float(long_rets.std())
            z = rets[-1] / ret_std if ret_std > 0 else 0.0
            signed = volume * math.erf(z / math.sqrt(2.0))  # == volume * (2 * Phi(z) - 1)
        self._volumes.append(volume)
        self._signed_volumes.append(signed)
        total_volume = sum(self._volumes)
        imbalance_1 = signed / volume if volume > 0 else 0.0
        imbalance_short = sum(self._signed_volumes) / total_volume if total_volume > 0 else 0.0

        x_ret = np.concatenate(
            [
                rets[::-1][: self.n_lags],
                [
                    short_rets.mean(),
                    long_rets.mean(),
                    long_rets.std(),
                    rsi,
                    bar.close / self._ema - 1.0,
                    (bar.high - bar.low) / bar.close,
                    volume_z,
                    imbalance_1,
                    imbalance_short,
                ],
            ]
        )

        rv_long = math.sqrt(float(np.mean(long_rets**2)))
        floor = max(0.1 * rv_long, 1e-10)

        def log_vol(value: float) -> float:
            return math.log(max(value, floor))

        day_fraction = (bar.timestamp.hour * 3600 + bar.timestamp.minute * 60) / 86400.0
        angle = 2.0 * math.pi * day_fraction
        x_vol = np.array(
            [
                log_vol(abs(rets[-1])),
                log_vol(math.sqrt(float(np.mean(short_rets**2)))),
                log_vol(rv_long),
                log_vol(_PARKINSON * self._log_ranges[-1]),
                log_vol(_PARKINSON * float(np.mean(self._log_ranges))),
                volume_z,
                math.sin(angle),
                math.cos(angle),
                math.sin(2 * angle),
                math.cos(2 * angle),
            ]
        )
        return FeatureSet(
            returns=np.nan_to_num(x_ret, nan=0.0, posinf=0.0, neginf=0.0),
            volatility=np.nan_to_num(x_vol, nan=0.0, posinf=0.0, neginf=0.0),
            baseline_vol=rv_long,
        )
