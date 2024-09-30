from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

API_KEY = "#####"
SECRET_KEY = "#####"

trading_client = TradingClient(API_KEY, SECRET_KEY)
# Get the last 100 closed orders
get_orders_data = GetOrdersRequest(
    status=QueryOrderStatus.CLOSED,
    limit=100,
    nested=True  # show nested multi-leg orders
)

trading_client.get_orders(filter=get_orders_data)