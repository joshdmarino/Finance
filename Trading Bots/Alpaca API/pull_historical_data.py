# Pulls historical data at specified time for specified ticker(s)

from alpaca.data import StockHistoricalDataClient, StockTradesRequest
from datetime import datetime

API_KEY = "#####"
SECRET_KEY = "#####"

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request_params = StockTradesRequest(
    symbol_or_symbols = "AAPL",
    start = datetime(2024 ,1, 30, 14, 30),
    end = datetime(2024, 1, 30, 14, 45)
)

trades = data_client.get_stock_trades(request_params)
print(trades)