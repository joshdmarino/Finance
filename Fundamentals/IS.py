import pandas as pd
import requests

ticker = ['XOM']

api = 'IwvL1bet534L4QH5HLKJPtgbMpwzDtzY'

URL = 'https://financialmodelingprep.com/api/v3/'
IS = 'income-statement/'
ticker = ticker[0]
period = 'annual'
limit = 3
r = requests.get('{}{}{}?period={}&limit={}&apikey={}'.format(URL,IS,ticker,period,limit,api))


d = pd.DataFrame.from_dict(r.json()).transpose()

d.to_excel('fundamentals.xlsx')