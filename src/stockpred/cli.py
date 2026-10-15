"""Command-line entry point: `stockpred stream`, `stockpred backtest` and `stockpred serve`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from datetime import timedelta
from pathlib import Path

from stockpred.models import MODELS
from stockpred.pipeline import ForecastPipeline
from stockpred.sources import (
    BarSource,
    FinnhubSource,
    MergedSource,
    ReplaySource,
    SimulatedSource,
    YFinancePollingSource,
    interleave,
    load_history,
)
from stockpred.types import Forecast

_INTERVAL_RE = re.compile(r"^(\d+)([smhd])$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Set KEY=VALUE pairs from a .env file; variables already in the environment win."""
    env_file = Path(path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_interval(text: str) -> timedelta:
    """Parse intervals like '30s', '1m', '5m', '1h' into a timedelta."""
    match = _INTERVAL_RE.match(text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid interval {text!r} (expected e.g. 30s, 1m, 1h)")
    return timedelta(**{_UNITS[match.group(2)]: int(match.group(1))})


def format_forecast(f: Forecast) -> str:
    band = f" [{f.lower:.2f}, {f.upper:.2f}]" if f.lower is not None else ""
    signal = f.direction if f.signal_active else f"flat (raw {f.raw_predicted_return:+.4%})"
    return (
        f"{f.timestamp:%Y-%m-%d %H:%M:%S}  {f.symbol:<10} close={f.last_close:<10.2f} "
        f"h={f.horizon}  vol={f.predicted_volatility:.3%}  "
        f"pred={f.predicted_return:+.4%} -> {f.predicted_price:.2f}{band}  {signal}"
    )


def _split(text: str | None) -> list[str]:
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def _symbols(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """All symbols to fetch (references first) and the reference symbols."""
    refs = _split(args.reference)
    return list(dict.fromkeys(refs + _split(args.symbol))), refs


def build_source(args: argparse.Namespace) -> BarSource:
    symbols, refs = _symbols(args)
    replay = args.command == "backtest"
    if args.source == "sim":
        sims = [
            SimulatedSource(
                symbol=symbol,
                n_bars=args.bars,
                interval=parse_interval(args.interval),
                seed=None if args.seed is None else args.seed + i,
                delay=0.0 if replay else args.delay,
            )
            for i, symbol in enumerate(symbols)
        ]
        if replay:
            return ReplaySource(interleave([list(s.bars()) for s in sims], refs))
        return sims[0] if len(sims) == 1 else MergedSource(sims)
    if args.source == "finnhub":
        return FinnhubSource(
            symbols, os.environ.get("FINNHUB_API_KEY", ""), parse_interval(args.interval)
        )
    if args.source == "yfinance":
        if replay:
            histories = [load_history(s, args.period, args.interval) for s in symbols]
            return ReplaySource(interleave(histories, refs))
        pollers = [YFinancePollingSource(s, args.interval, args.poll_seconds) for s in symbols]
        return pollers[0] if len(pollers) == 1 else MergedSource(pollers)
    raise SystemExit(f"unknown source {args.source!r}")


def build_pipeline(args: argparse.Namespace) -> ForecastPipeline:
    pipeline = ForecastPipeline(
        MODELS[args.model],
        horizon=args.horizon,
        gate=not args.no_gate,
        coverage=args.coverage,
        reference_symbols=_split(args.reference),
    )
    if args.state and Path(args.state).exists():
        try:
            pipeline.load(args.state)
        except ValueError as exc:
            raise SystemExit(f"cannot resume from {args.state}: {exc}") from exc
        logging.getLogger(__name__).info(
            "resumed state for %s from %s", ", ".join(pipeline.symbols), args.state
        )
    return pipeline


def _print_metrics(pipeline: ForecastPipeline) -> None:
    for symbol in pipeline.symbols:
        print(f"[{symbol}]\n{pipeline.metrics(symbol).summary()}")


async def _run(pipeline: ForecastPipeline, source: BarSource, args: argparse.Namespace) -> None:
    count = 0
    async for forecast in pipeline.run(source):
        count += 1
        if not args.quiet:
            print(format_forecast(forecast))
        if args.report_every and count % args.report_every == 0:
            _print_metrics(pipeline)
        if args.state and args.save_every and count % args.save_every == 0:
            pipeline.save(args.state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockpred", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", default="AAPL", help="symbol, or comma-separated list")
    common.add_argument(
        "--reference", default="", help="reference symbol(s) for cross-asset features, e.g. SPY"
    )
    common.add_argument("--model", choices=sorted(MODELS), default="sgd")
    common.add_argument("--horizon", type=int, default=1, help="forecast horizon in bars")
    common.add_argument(
        "--no-gate", action="store_true", help="serve return forecasts even without recent skill"
    )
    common.add_argument("--coverage", type=float, default=0.95, help="forecast band coverage")
    common.add_argument("--interval", default="1m", help="bar interval, e.g. 1m, 5m")
    common.add_argument("--seed", type=int, default=None, help="sim source RNG seed")
    common.add_argument("--state", default=None, help="file to resume learned state from/save to")
    common.add_argument("--save-every", type=int, default=500, help="save state every N forecasts")
    common.add_argument("-v", "--verbose", action="store_true")

    live = argparse.ArgumentParser(add_help=False)
    live.add_argument("--source", choices=["sim", "yfinance", "finnhub"], default="sim")
    live.add_argument("--poll-seconds", type=float, default=30.0)
    live.add_argument("--bars", type=int, default=None, help="sim: stop after N bars")
    live.add_argument("--delay", type=float, default=0.5, help="sim: seconds per bar")

    printing = argparse.ArgumentParser(add_help=False)
    printing.add_argument("--report-every", type=int, default=100, help="0 to disable")
    printing.add_argument("--quiet", action="store_true", help="hide per-bar forecasts")

    sub.add_parser(
        "stream", parents=[common, live, printing], help="print forecasts from a live stream"
    )

    backtest = sub.add_parser(
        "backtest", parents=[common, printing], help="replay history as a stream"
    )
    backtest.add_argument("--source", choices=["sim", "yfinance"], default="yfinance")
    backtest.add_argument("--period", default="5d", help="yfinance history period")
    backtest.add_argument("--bars", type=int, default=5000, help="sim: number of bars")
    backtest.set_defaults(quiet=None)

    serve = sub.add_parser("serve", parents=[common, live], help="run the live API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--history", type=int, default=500, help="forecasts kept per symbol")
    return parser


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from stockpred.server import ForecastService, create_app

    symbols, _ = _symbols(args)
    service = ForecastService(
        build_pipeline(args),
        build_source(args),
        source_name=f"{args.source}: {', '.join(symbols)} ({args.interval} bars)",
        history=args.history,
        state_path=args.state,
        save_every=args.save_every,
    )
    print(f"dashboard: http://{args.host}:{args.port}/")
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "serve":
        _serve(args)
        return
    if args.quiet is None:  # backtests are quiet by default
        args.quiet = True

    pipeline = build_pipeline(args)
    source = build_source(args)
    try:
        asyncio.run(_run(pipeline, source, args))
    except KeyboardInterrupt:
        print("\ninterrupted")
    if args.state:
        pipeline.save(args.state)
    print("final metrics:")
    _print_metrics(pipeline)


if __name__ == "__main__":
    main()
