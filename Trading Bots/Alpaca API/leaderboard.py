# Example code to track members' portfolios. requires them sharing keys but should prevent cheating as a change in API keys would mean a change in account settings.

from alpaca.trading.client import TradingClient
import pandas as pd


leaderboard = pd.DataFrame({'Name': [],'Account Value':  [],})

name1 = "Jake"
API_KEY = "#####"
SECRET_KEY = "#####"
name1_account = TradingClient(API_KEY, SECRET_KEY)
name1_account_value = name1_account.get_account().portfolio_value
leaderboard.loc[0] = [name1,name1_account_value]

name2 = "Josh"
API_KEY = "#####"
SECRET_KEY = "#####"
name2_account = TradingClient(API_KEY, SECRET_KEY)
name2_account_value = name2_account.get_account().portfolio_value
leaderboard.loc[1] = [name2,name2_account_value]


name3 = "Pranav"
API_KEY = "#####"
SECRET_KEY = "#####"
name3_account = TradingClient(API_KEY, SECRET_KEY)
name3_account_value = name3_account.get_account().portfolio_value
leaderboard.loc[2] = [name3,name3_account_value]

leaderboard['Rank'] = leaderboard['Account Value'].rank(ascending=False).astype(int)
leaderboard = leaderboard[['Rank', 'Name', 'Account Value']].sort_values(by='Rank')
print (leaderboard.to_string(index=False))