# Runs limit order for Alpaca API. Insert your own keys

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = "#####"
SECRET_KEY = "#####"

trading_client = TradingClient(API_KEY, SECRET_KEY)
limit_order_data = LimitOrderRequest(
    symbol = "SPY",
    qty = 1,
    side = OrderSide.BUY,
    time_in_force = TimeInForce.DAY,
    limit_price = 486.00
)

limit_order = trading_client.submit_order(limit_order_data)
print(limit_order)