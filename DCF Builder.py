import pandas as pd
import requests
from api_keys import fmp_key, API_KEY, SECRET_KEY
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
from io import BytesIO
import warnings
from urllib3.exceptions import InsecureRequestWarning
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Suppress only the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# Setup the parameters
ticker = 'AAPL'
URL = 'https://financialmodelingprep.com/api/v3/'
period = 'annual'
limit = 4  # Get 4 years of data to calculate 3 changes in NWC

# Download BS, IS, CFS from FMP data
BalanceSheet = requests.get(f'{URL}balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={fmp_key}').json()
IncomeStat = requests.get(f'{URL}income-statement/{ticker}?period={period}&limit={limit}&apikey={fmp_key}').json()
CashFlowStat = requests.get(f'{URL}cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={fmp_key}').json()

# Create an empty list to store each year's data
data = []

# Variables to hold cumulative sums for averages
revenue_growth_sum = 0
ebitda_margin_sum = 0
ebit_percent_sum = 0
tax_rate_sum = 0
nwc_change_sum = 0
capex_percent_sum = 0

# Loop through 3 years of data for calculations (using 4 years to calculate changes)
for i in range(3):  # Loop for 3 years
    revenue = IncomeStat[i]['revenue']
    revenue_growth = (IncomeStat[i]['revenue'] - IncomeStat[i+1]['revenue']) / IncomeStat[i+1]['revenue'] if i < 2 else None
    ebitda = IncomeStat[i]['ebitda']
    ebitda_margin = ebitda / revenue
    ebit = IncomeStat[i]['operatingIncome']
    ebit_percent_of_revenue = ebit / revenue
    tax_rate = IncomeStat[i]['incomeTaxExpense'] / IncomeStat[i]['incomeBeforeTax'] if IncomeStat[i]['incomeBeforeTax'] != 0 else 0
    ebiat = ebit - (ebit * tax_rate)
    dep_amor = ebitda - ebit

    # Calculate Net Working Capital and the change from the previous year
    nwc = BalanceSheet[i]['totalCurrentAssets'] - BalanceSheet[i]['totalCurrentLiabilities']
    change_in_nwc = (nwc - (BalanceSheet[i-1]['totalCurrentAssets'] - BalanceSheet[i-1]['totalCurrentLiabilities'])) if i < 2 else None

    capex = CashFlowStat[i]['capitalExpenditure']
    unlevered_cash_from_operations = sum([ebiat, dep_amor, change_in_nwc]) if change_in_nwc is not None else None
    unlevered_fcf = unlevered_cash_from_operations - abs(capex) if unlevered_cash_from_operations is not None else None
    cash_and_equivalents = BalanceSheet[i]['cashAndCashEquivalents']
    long_term_debt = BalanceSheet[i]['longTermDebt']
    net_debt = long_term_debt - cash_and_equivalents
    capex_percent_of_revenue = capex / revenue

    # Accumulate for averages
    revenue_growth_sum += revenue_growth if revenue_growth is not None else 0
    ebitda_margin_sum += ebitda_margin
    ebit_percent_sum += ebit_percent_of_revenue
    tax_rate_sum += tax_rate
    nwc_change_sum += change_in_nwc if change_in_nwc is not None else 0
    capex_percent_sum += capex_percent_of_revenue

    # Append data to the list
    data.append({
        'Year': IncomeStat[i]['date'],
        '': " ",
        'Revenue': revenue,
        'Revenue Growth': revenue_growth,
        'EBITDA': ebitda,
        'EBITDA Margin': ebitda_margin,
        'EBIT': ebit,
        'EBIT Percent of Revenue': ebit_percent_of_revenue,
        'Tax Rate': tax_rate,
        'EBIAT': ebiat,
        ' ': " ",
        'Depreciation & Amortization': dep_amor,
        'Change in NWC': change_in_nwc,
        'Capex': capex,
        'Unlevered FCF': unlevered_fcf,
        '   ': ' ',
    })

# Calculate straight-line averages for the forecasts
avg_revenue_growth = revenue_growth_sum / 2  # Average of the 2 years of growth
avg_ebitda_margin = ebitda_margin_sum / 3
avg_ebit_percent = ebit_percent_sum / 3
avg_tax_rate = tax_rate_sum / 3
avg_nwc_change = nwc_change_sum / 2
avg_capex_percent = capex_percent_sum / 3

# Get the latest year of revenue for the forecast
last_year_revenue = IncomeStat[0]['revenue']

# Forecast 5 years into the future
for j in range(1, 6):  # Forecast for 5 years
    future_revenue = last_year_revenue * (1 + avg_revenue_growth)
    future_ebitda = future_revenue * avg_ebitda_margin
    future_ebit = future_revenue * avg_ebit_percent
    future_ebiat = future_ebit - (future_ebit * avg_tax_rate)
    future_dep_amor = future_ebitda - future_ebit
    future_nwc_change = avg_nwc_change
    future_capex = future_revenue * avg_capex_percent
    future_unlevered_fcf = future_ebiat + future_dep_amor - future_nwc_change - abs(future_capex)



    # Append forecast data
    data.append({
        'Year': f"Forecast Year {j}",
        '': " ",
        'Revenue': future_revenue,
        'Revenue Growth': avg_revenue_growth,
        'EBITDA': future_ebitda,
        'EBITDA Margin': avg_ebitda_margin,
        'EBIT': future_ebit,
        'EBIT Percent of Revenue': avg_ebit_percent,
        'Tax Rate': avg_tax_rate,
        'EBIAT': future_ebiat,
        ' ': " ",
        'Depreciation & Amortization': future_dep_amor,
        'Change in NWC': future_nwc_change,
        'Capex': future_capex,
        'Unlevered FCF': future_unlevered_fcf,
        '   ': ' ',
    })

    # Update last year revenue for the next forecast
    last_year_revenue = future_revenue

# Calculate the WACC

# get Rf
result = requests.get("https://www.cnbc.com/quotes/US10Y")
src = result.content
soup = BeautifulSoup(src, 'lxml')
Rf = float((soup.find(lambda tag: tag.name == 'span' and tag.get('class') == ['QuoteStrip-lastPrice'])).text[0:4])

# get ERP
excel_url = "https://www.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls"
response = requests.get(excel_url, verify=False)
if response.status_code == 200:
    df_ERP = pd.read_excel(BytesIO(response.content), skiprows=6)
    df_ERP = df_ERP.dropna(subset=["Year"]).iloc[:-1, :].set_index("Year")
    ERP = (df_ERP["Implied ERP (FCFE)"].values[-1])*100
   

# get beta
info = yf.Ticker(ticker).info
beta = info['beta']

# Calculate cost of equity
CostOfEquity = beta * (ERP) + Rf

# get cost of debt (fix later)
interest_expense = IncomeStat[0]['interestExpense']
total_debt = BalanceSheet[0]['totalDebt']
cost_of_debt = ((interest_expense / total_debt) * (1 - tax_rate))*150

# calculate WACC
Assets = BalanceSheet[0]["totalAssets"]
Debt = BalanceSheet[0]["totalLiabilities"]
marketcap = info['marketCap']
total = marketcap + Debt
AfterTaxCostOfDebt = cost_of_debt * (1 - avg_tax_rate)
WACC = ((AfterTaxCostOfDebt * Debt / total) + (CostOfEquity * marketcap / total))/100

# Convert the list to a DataFrame for display
df = pd.DataFrame(data)

# Calculate Present Value of Unlevered Free Cash Flows using WACC
discounted_fcf = []
for j in range(1, 6):  # Forecasted years
    forecasted_fcf = df.loc[df['Year'] == f"Forecast Year {j}", 'Unlevered FCF'].values[0]
    
    # Calculate the present value of each forecasted FCF
    present_value_fcf = forecasted_fcf / ((1 + WACC) ** j)
    discounted_fcf.append(present_value_fcf)
    
    # Add the present value FCF to the DataFrame
    df.loc[df['Year'] == f"Forecast Year {j}", 'Present Value of Unlevered FCF'] = present_value_fcf

# Summing up the discounted unlevered FCF
total_pv_fcf = sum(discounted_fcf)

# Perpetual growth rate
g = 0.02  # 2%

# Get the final year's FCF for terminal value calculation
final_year_fcf = df.loc[df['Year'] == "Forecast Year 5", 'Unlevered FCF'].values[0]

# Calculate the Terminal Value (TV)
terminal_value = (final_year_fcf * (1 + g)) / (WACC - g)

# Discount the terminal value to present value
present_value_tv = terminal_value / ((1 + WACC) ** 5)

# Calculate the Enterprise Value (EV)
enterprise_value = total_pv_fcf + present_value_tv

# Subtract total debt and add cash and cash equivalents to get Equity Value
cash_and_equivalents = BalanceSheet[0]['cashAndCashEquivalents']
total_debt = BalanceSheet[0]['totalDebt']
equity_value = enterprise_value - net_debt + cash_and_equivalents

# Divide by diluted shares outstanding to get the implied share price
diluted_shares_outstanding = info['sharesOutstanding']
share_price = equity_value / diluted_shares_outstanding


# Export the DataFrame to an Excel file
df.transpose().to_excel("outputs.xlsx", index=False)


#Print the calculated share price
print(f"The implied share price is: ${share_price:.2f}")

ticker_yahoo = yf.Ticker(ticker)
data = ticker_yahoo.history()
last_quote = data['Close'].iloc[-1]
print(last_quote)

if last_quote < share_price and (1-last_quote/share_price) < .35:
    trading_client = TradingClient(API_KEY, SECRET_KEY)
    market_order_data = MarketOrderRequest(
    symbol = ticker,
    qty = 1,
    side = OrderSide.BUY,
    time_in_force=TimeInForce.GTC
)
market_order = trading_client.submit_order(market_order_data)
print(market_order)