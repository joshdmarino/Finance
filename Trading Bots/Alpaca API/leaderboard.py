# Example code to track members' portfolios. requires them sharing keys but should prevent cheating as a change in API keys would mean a change in account settings.

from alpaca.trading.client import TradingClient
import pandas as pd


leaderboard = pd.DataFrame({'Name': [],'Account Value':  [],})

name1 = "Jason"
API_KEY = 'PKZ1D2P6HZYHHHTAHSZP'
SECRET_KEY = 'I6o24rgo8ojurHESYTfdvcYLjvq2rXje2lV5Q5PC'
name1_account = TradingClient(API_KEY, SECRET_KEY)
name1_account_value = name1_account.get_account().portfolio_value
leaderboard.loc[0] = [name1,name1_account_value]

name2 = "Josh"
API_KEY = 'PKNC2EMD8OW41W3EFC5R'
SECRET_KEY = 'yW8AJSj0DQuYbG1XI7Ylr2kIcat8N6ouK5ADESQf'
name2_account = TradingClient(API_KEY, SECRET_KEY)
name2_account_value = name2_account.get_account().portfolio_value
leaderboard.loc[1] = [name2,name2_account_value]


name3 = "Caiden"
API_KEY = 'PK2BKC1RCUCJ677DMH6J'
SECRET_KEY = 'LI5g4dJc55YqaeOX1CNgHPwpJeFhnQPC3riPZf0b'
name3_account = TradingClient(API_KEY, SECRET_KEY)
name3_account_value = name3_account.get_account().portfolio_value
leaderboard.loc[2] = [name3,name3_account_value]

name4 = "Aditya"
API_KEY = 'PKOX4U0AX4HVVCZGCMPU'
SECRET_KEY = 'UaViCYBKf246E9wiPpcNAZJXIieisHnftQK8NacC'
name4_account = TradingClient(API_KEY, SECRET_KEY)
name4_account_value = name4_account.get_account().portfolio_value
leaderboard.loc[3] = [name4,name4_account_value]

# name5 = "Jack"
# API_KEY = 'PK51XGDWDEWDNN7WRIIG'
# SECRET_KEY = 'hC4YsqH6RkhSCBYhSDGPk7oc5BZ5HGJbil55oygm'
# name5_account = TradingClient(API_KEY, SECRET_KEY)
# name5_account_value = name5_account.get_account().portfolio_value
# leaderboard.loc[4] = [name5,name5_account_value]

name6 = "Nathan"
API_KEY = 'PKAOLWRYAB8JN4JP021H'
SECRET_KEY = '3gvYO5X4aJg3AyracrdrZVo4MsxGGZoaexBww1j5'
name6_account = TradingClient(API_KEY, SECRET_KEY)
name6_account_value = name6_account.get_account().portfolio_value
leaderboard.loc[5] = [name6,name6_account_value]

leaderboard['Rank'] = leaderboard['Account Value'].rank(ascending=False).astype(int)
leaderboard = leaderboard[['Rank', 'Name', 'Account Value']].sort_values(by='Rank')
print (leaderboard.to_string(index=False))