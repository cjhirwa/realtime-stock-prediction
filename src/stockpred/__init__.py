"""Real-time stock trend forecasting over streaming market data."""

from stockpred.pipeline import ForecastPipeline
from stockpred.types import Bar, Forecast, Trade

__all__ = ["Bar", "Forecast", "ForecastPipeline", "Trade"]
__version__ = "0.1.0"
