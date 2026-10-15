"""Live forecast service: runs the pipeline in the background and serves it over HTTP and
WebSocket, with a dashboard at `/`."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from stockpred.pipeline import ForecastPipeline
from stockpred.sources import BarSource
from stockpred.types import Forecast

log = logging.getLogger(__name__)


class ForecastService:
    """Owns a pipeline and its source; keeps recent forecasts and fans them out to clients."""

    def __init__(
        self,
        pipeline: ForecastPipeline,
        source: BarSource,
        source_name: str = "",
        history: int = 500,
        state_path: str | os.PathLike[str] | None = None,
        save_every: int = 500,
        subscriber_queue_size: int = 256,
    ) -> None:
        self.pipeline = pipeline
        self.source = source
        self.source_name = source_name
        self.history_size = history
        self.state_path = state_path
        self.save_every = save_every
        self.subscriber_queue_size = subscriber_queue_size
        self.history: dict[str, deque[Forecast]] = {}
        self.forecast_count = 0
        self.status = "starting"
        self.error: str | None = None
        self.started_at = datetime.now(UTC)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def run(self) -> None:
        self.status = "running"
        try:
            async for forecast in self.pipeline.run(self.source):
                self._record(forecast)
            self.status = "finished"
        except asyncio.CancelledError:
            self.status = "stopped"
            raise
        except Exception as exc:
            log.exception("forecast stream failed")
            self.status = "error"
            self.error = str(exc)
        finally:
            self.save()

    def _record(self, forecast: Forecast) -> None:
        self.history.setdefault(forecast.symbol, deque(maxlen=self.history_size)).append(forecast)
        self.forecast_count += 1
        message = {"type": "forecast", "forecast": forecast.as_dict()}
        for queue in list(self._subscribers):
            with suppress(asyncio.QueueFull):  # a slow client misses updates, not the stream
                queue.put_nowait(message)
        if self.save_every and self.forecast_count % self.save_every == 0:
            self.save()

    def save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.pipeline.save(self.state_path)
        except OSError:
            log.exception("could not save pipeline state to %s", self.state_path)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self.subscriber_queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def status_dict(self) -> dict[str, Any]:
        p = self.pipeline
        return {
            "status": self.status,
            "error": self.error,
            "source": self.source_name,
            "model": p.model_factory().name,
            "horizon": p.horizon,
            "gate": p.gate,
            "coverage": p.coverage,
            "reference_symbols": p.reference_symbols,
            "symbols": sorted(self.history),
            "forecast_count": self.forecast_count,
            "started_at": self.started_at.isoformat(),
        }


def create_app(service: ForecastService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(service.run())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    app = FastAPI(title="Real-Time Stock Prediction", lifespan=lifespan)
    dashboard = files("stockpred").joinpath("static/dashboard.html").read_text(encoding="utf-8")

    def known(symbol: str) -> str:
        if symbol not in service.history:
            raise HTTPException(status_code=404, detail=f"no forecasts for {symbol!r}")
        return symbol

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return dashboard

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return service.status_dict()

    @app.get("/api/symbols")
    async def symbols() -> list[str]:
        return sorted(service.history)

    @app.get("/api/forecasts/{symbol}")
    async def forecasts(
        symbol: str, limit: int = Query(300, ge=1, le=5000)
    ) -> list[dict[str, Any]]:
        history = service.history[known(symbol)]
        return [f.as_dict() for f in list(history)[-limit:]]

    @app.get("/api/metrics")
    async def all_metrics() -> dict[str, Any]:
        return {s: service.pipeline.metrics(s).as_dict() for s in sorted(service.history)}

    @app.get("/api/metrics/{symbol}")
    async def metrics(symbol: str) -> dict[str, Any]:
        return service.pipeline.metrics(known(symbol)).as_dict()

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = service.subscribe()
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            service.unsubscribe(queue)

    return app
