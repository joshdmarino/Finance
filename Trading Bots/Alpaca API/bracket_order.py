from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

API_KEY = "#####"
SECRET_KEY = "#####"

trading_client = TradingClient(API_KEY, SECRET_KEY)

# preparing bracket order with both stop loss and take profit
bracket__order_data = MarketOrderRequest(
                    symbol="SPY",
                    qty=5,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=400),
                    stop_loss=StopLossRequest(stop_price=300)
                    )

bracket_order = trading_client.submit_order(
                order_data=bracket__order_data
               )

# preparing oto order with stop loss
oto_order_data = LimitOrderRequest(
                    symbol="SPY",
                    qty=5,
                    limit_price=350,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_loss=StopLossRequest(stop_price=300)
                    )

# Market order
oto_order = trading_client.submit_order(
                order_data=oto_order_data
               )