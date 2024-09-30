import requests
import pandas as pd
import time
from datetime import datetime
from api_keys import API_KEY, SECRET_KEY

# Constants
BASE_URL = 'https://data.alpaca.markets/v1beta3/crypto/us'  # Updated for v1beta3 API
ASSET = 'BTC/USD'  # Format of the trading pair (use BTC/USD for crypto)
TIMEFRAME = '1Min'  # Timeframe for bar data
SHORT_WINDOW = 5  # Short moving average window
LONG_WINDOW = 20  # Long moving average window
QTY_TO_TRADE = 0.01  # Quantity of crypto to trade

# Headers for Alpaca API authentication
HEADERS = {
    'Apca-Api-Key-Id': API_KEY,
    'Apca-Api-Secret-Key': SECRET_KEY
}

def get_crypto_data(symbol, timeframe, limit=100):
    """
    Retrieve historical crypto data using the updated v1beta3 API.
    """
    try:
        # Construct the request URL using the correct symbol format and the /v1beta3 endpoint
        url = f"{BASE_URL}/bars?timeframe={timeframe}&symbols={symbol}&limit={limit}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        # Parse response JSON
        data = response.json()
        if symbol not in data['bars']:
            raise ValueError(f"No data found for {symbol}")

        # Extract bars and convert to DataFrame
        bars = data['bars'][symbol]
        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['t'])
        df.set_index('time', inplace=True)
        return df[['o', 'h', 'l', 'c', 'v']]  # Return relevant columns: open, high, low, close, volume
    
    except Exception as e:
        print(f"Error retrieving crypto data: {e}")
        return None

def calculate_moving_averages(data, short_window, long_window):
    """
    Calculate short and long moving averages.
    """
    data['SMA_short'] = data['c'].rolling(window=short_window, min_periods=1).mean()
    data['SMA_long'] = data['c'].rolling(window=long_window, min_periods=1).mean()
    return data

def place_order(symbol, qty, side):
    """
    Place a market order to buy or sell the specified quantity of crypto.
    """
    # Alpaca crypto trading functionality here (use the correct REST API method for trading)
    print(f"Placing {side} order for {qty} {symbol} (mock order).")

def trading_strategy(symbol, short_window, long_window, qty_to_trade):
    """
    Execute the moving average trading strategy.
    """
    # Get the most recent data
    data = get_crypto_data(symbol, TIMEFRAME)
    if data is None:
        return
    
    # Calculate moving averages
    data = calculate_moving_averages(data, short_window, long_window)

    # Get the most recent row (the latest price data)
    latest_data = data.iloc[-1]
    prev_data = data.iloc[-2]
    
    # Print data to help debug
    print("\nLatest Data:")
    print(data.tail(5))  # Print the last 5 rows, including moving averages
    print(f"\nSMA_short: {latest_data['SMA_short']} | SMA_long: {latest_data['SMA_long']}")

    # Check for crossover
    if prev_data['SMA_short'] < prev_data['SMA_long'] and latest_data['SMA_short'] > latest_data['SMA_long']:
        print("Bullish crossover detected. Placing BUY order.")
        place_order(symbol, qty_to_trade, 'buy')
    elif prev_data['SMA_short'] > prev_data['SMA_long'] and latest_data['SMA_short'] < latest_data['SMA_long']:
        print("Bearish crossover detected. Placing SELL order.")
        place_order(symbol, qty_to_trade, 'sell')
    else:
        print("No crossover detected. No action taken.")

if __name__ == "__main__":
    while True:
        print(f"\nRunning strategy at {datetime.now()}")
        trading_strategy(ASSET, SHORT_WINDOW, LONG_WINDOW, QTY_TO_TRADE)
        time.sleep(60)  # Pause for 1 minute before the next iteration
