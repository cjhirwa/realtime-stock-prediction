import time

from fastapi.testclient import TestClient

from stockpred.models import OnlineLinearForecaster
from stockpred.pipeline import ForecastPipeline
from stockpred.server import ForecastService, create_app
from stockpred.sources import SimulatedSource


def _client(tmp_path, n_bars: int, delay: float = 0.0) -> tuple[TestClient, ForecastService]:
    service = ForecastService(
        ForecastPipeline(OnlineLinearForecaster),
        SimulatedSource(n_bars=n_bars, seed=1, delay=delay),
        source_name="sim",
        history=100,
        state_path=tmp_path / "state.pkl",
        save_every=50,
    )
    return TestClient(create_app(service)), service


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError
        time.sleep(0.02)


def test_rest_endpoints(tmp_path):
    client, service = _client(tmp_path, n_bars=300)
    with client:
        _wait_for(lambda: service.status == "finished")

        status = client.get("/api/status").json()
        assert status["status"] == "finished" and status["symbols"] == ["SIM"]
        assert status["forecast_count"] == 300 - 21 + 1

        forecasts = client.get("/api/forecasts/SIM", params={"limit": 10}).json()
        assert len(forecasts) == 10
        assert {"timestamp", "predicted_volatility", "lower", "upper", "direction"} <= set(
            forecasts[-1]
        )
        assert client.get("/api/forecasts/NOPE").status_code == 404

        metrics = client.get("/api/metrics/SIM").json()
        assert metrics["volatility"]["n"] > 0
        assert set(client.get("/api/metrics").json()) == {"SIM"}

        page = client.get("/")
        assert page.status_code == 200 and "Real-Time Stock Prediction" in page.text
    assert (tmp_path / "state.pkl").exists()


def test_websocket_pushes_forecasts(tmp_path):
    client, _ = _client(tmp_path, n_bars=2000, delay=0.005)
    with client, client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
        assert message["type"] == "forecast"
        assert message["forecast"]["symbol"] == "SIM"
