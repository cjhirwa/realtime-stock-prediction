# Real-Time Stock Prediction

A machine learning system that forecasts price trends with time-series models over streaming data from financial APIs.

## How it works

```
 source ──bars──▶ FeatureBuilder ──x──▶ Forecaster.predict ──▶ Forecast
   │                                          ▲
   │      (h bars later)  realized return ────┴──▶ metrics + Forecaster.learn
```

- **Sources** ([sources.py](src/stockpred/sources.py)) emit OHLCV bars asynchronously:
  - `finnhub`: live trades over Finnhub's WebSocket, aggregated into bars ([aggregator.py](src/stockpred/aggregator.py)). Needs `FINNHUB_API_KEY`, set in the environment or in `.env`. On the free plan, crypto streams densely (about 40 BTC trades a second), but US stock trades are sparse (1 SPY trade in 30 s during market hours), so use `yfinance` for stocks.
  - `yfinance`: polls Yahoo Finance for completed bars. No key needed; data is delayed.
  - `sim`: synthetic AR(1) returns with a known signal, for development and tests.
- **Features** ([features.py](src/stockpred/features.py)) are computed incrementally per symbol and are scale-free:
  - *Return features:* lagged log returns, rolling mean and volatility, RSI, distance from EMA, bar range, volume z-score, and order-flow imbalance. Imbalance comes from tick-rule signed volume for Finnhub trades, or is estimated from each bar with bulk volume classification for bar-only feeds.
  - *Cross-asset features (opt-in, `--reference SPY`):* the reference symbol's latest, previous and 5-bar mean return as of each bar, plus the symbol's return relative to it.
  - *Volatility features:* log realized volatility over 1, 5 and 20 bars (HAR), Parkinson high-low volatility, volume, and time-of-day harmonics.
- **Return models** ([models.py](src/stockpred/models.py)) predict the log return over the next `h` bars:
  - `sgd`: online linear regression (an adaptive AR model with extra features) with Huber loss, using running feature and target scaling.
  - `ewma`, `naive`: baselines.
- **Skill gate:** a return forecast is served only while the model's skill over its last 300 scored forecasts is positive. Otherwise the served forecast is 0 (flat), while the model keeps learning in the background. Forecasts also carry the ungated `raw_predicted_return`.
- **Volatility model** (`har`) forecasts realized volatility over the horizon. It's an online regression on log volatility, rescaled online to minimize QLIKE.
- **Forecast band:** `predicted_return ± q · predicted_volatility`, where `q` is calibrated from recent normalized errors (conformal). That way the 95% band actually contains about 95% of outcomes.
- **Pipeline** ([pipeline.py](src/stockpred/pipeline.py)) works test-then-train. Each forecast is scored once its horizon passes, and only then do the models learn from it. So all reported metrics are honest out-of-sample numbers, and the models keep adapting to regime changes.
- **Metrics:**
  - *Returns:* MAE, RMSE, directional accuracy, active share (how often a non-zero call is served), and **skill** = 1 − MAE / MAE of the random walk. These are reported for both served and raw forecasts.
  - *Volatility:* QLIKE, skill against the 20-bar historical-volatility baseline, and band coverage.

## Results

These are out-of-sample backtests over AAPL, MSFT, SPY, TSLA and NVDA, on 1-minute bars (5 days) and 5-minute bars (60 days):

| | Return direction | Volatility |
|---|---|---|
| Raw model | Skill −0.02 to −0.12, ~50% direction: **no edge** | |
| Gated (served) | Skill ≈ 0 (−0.018 to +0.000); calls made 0–22% of the time | |
| HAR model against historical vol | | Skill **+0.41 to +0.75** on 5m bars, +0.15 to +0.31 on 1m bars (one exception: MSFT 1m h=5, −0.07) |
| Band coverage (95% target) | | 93.7%–95.7% |

Minute-level price direction is close to unpredictable from price and volume alone, and the gate correctly stays mostly closed. Volatility is where the predictive value is: the forecast bands are both tighter than the baseline's and honestly calibrated.

I tested the extra direction features on the same data. Neither helped:
- **SPY as reference:** lowered raw direction skill by about 0.01 in every setup. Extra features without signal add noise to an online model.
- **Bar-based order-flow proxy:** no measurable gain.

Tick-rule order flow from live Finnhub trades is untested, because there's no trade history to backtest it. The cross-asset mechanism itself is verified on synthetic lead-lag data ([test_cross_asset.py](tests/test_cross_asset.py)).

## Live dashboard and API

`stockpred serve` runs the pipeline in the background and serves:

| Route | |
|---|---|
| `GET /` | Live dashboard: price, calibrated band at the target time, signal, and per-symbol skill/coverage |
| `GET /api/status` | Source, model, horizon, symbols, stream status |
| `GET /api/forecasts/{symbol}?limit=300` | Recent forecasts |
| `GET /api/metrics[/{symbol}]` | Out-of-sample return and volatility metrics |
| `WS /ws` | Pushes each new forecast as `{"type": "forecast", "forecast": {...}}` |

## Saving and resuming state

Pass `--state FILE` to any command to resume learned state (features, models, gates, metrics) from that file and to save back to it. It saves every `--save-every` forecasts and on exit. Unresolved forecasts are dropped on save, since their horizons would span the downtime. The file is a pickle, so only load files this program wrote.

## Setup

```bash
uv sync
```

## Usage

```bash
# Live dashboard at http://127.0.0.1:8000 (synthetic data, no network)
uv run stockpred serve --source sim --symbol "ALPHA,BETA" --delay 0.2

# Live dashboard on real data, with SPY as a cross-asset reference, resuming state
uv run stockpred serve --source yfinance --symbol "AAPL,MSFT" --reference SPY --state data/state.pkl

# Synthetic live stream in the terminal
uv run stockpred stream --source sim --delay 0.2

# Backtest on the last 5 days of 1-minute bars, predicting 5 bars ahead
uv run stockpred backtest --symbol AAPL --period 5d --interval 1m --horizon 5

# Compare against the baseline
uv run stockpred backtest --symbol AAPL --model naive

# Poll Yahoo Finance during market hours
uv run stockpred stream --source yfinance --symbol MSFT

# True streaming via Finnhub (key in .env); crypto trades 24/7
uv run stockpred serve --source finnhub --symbol "BINANCE:BTCUSDT,BINANCE:ETHUSDT" --interval 10s
```

## Development

```bash
uv run pytest
uv run ruff check .
```

> Not financial advice. Short-horizon returns are mostly noise; expect return skill close to zero on real data.
