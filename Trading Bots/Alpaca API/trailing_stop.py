from alpaca.trading.client import TradingClient
from alpaca.trading.requests import TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = "#####"
SECRET_KEY = "#####"

trading_client = TradingClient(API_KEY, SECRET_KEY)

trailing_percent_data = TrailingStopOrderRequest(
                    symbol="SPY",
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=1.00 # hwm * 0.99
                    )

trailing_percent_order = trading_client.submit_order(
                order_data=trailing_percent_data
               )


trailing_price_data = TrailingStopOrderRequest(
                    symbol="SPY",
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_price=1.00 # hwm - $1.00
                    )

trailing_price_order = trading_client.submit_order(
                order_data=trailing_price_data
               )