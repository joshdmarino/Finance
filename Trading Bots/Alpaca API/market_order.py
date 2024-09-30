# Runs market order for Alpaca API. Insert your own keys

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = 'PKNC2EMD8OW41W3EFC5R'
SECRET_KEY = 'yW8AJSj0DQuYbG1XI7Ylr2kIcat8N6ouK5ADESQf'

trading_client = TradingClient(API_KEY, SECRET_KEY)
market_order_data = MarketOrderRequest(
    symbol = "BTCUSD",
    qty = .1,
    side = OrderSide.BUY,
    time_in_force=TimeInForce.GTC
)
market_order = trading_client.submit_order(market_order_data)
print(market_order)