import yfinance as yf
import numpy as np
import math
import pandas as pd
from scipy.stats import norm, rankdata
import pyvinecopulib as pvc
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Price Puts using Black-Scholes for Each Stock
# ------------------------------

tickers = ['AAPL', 'MSFT', 'GOOG']
T = 0.5  # Time to expiration in years
num_days = int(T * 252)  # Approximate number of trading days in T years

# Retrieve the risk-free rate from the 10-year treasury yield ("^TNX")
tnx = yf.Ticker("^TNX")
tnx_data = tnx.history(period="1d")
if tnx_data.empty:
    print("Error: Could not retrieve risk-free rate data from '^TNX'.")
    exit(1)
risk_free_rate = tnx_data['Close'].iloc[-1].item() / 100
r = risk_free_rate

# Dictionaries to store values for each stock
current_prices = {}
sigmas = {}
strike_prices = {}
put_prices = {}
# Also store daily arithmetic returns info for simulation
daily_means = {}
daily_variances = {}

for ticker in tickers:
    stock_data = yf.download(ticker, period="1y", auto_adjust=True)
    if stock_data.empty:
        print(f"Error: No data for {ticker}.")
        exit(1)
    
    prices = stock_data['Close']
    try:
        S = prices.iloc[-1].item()
    except IndexError:
        print(f"Error: Price series is empty for {ticker}.")
        exit(1)
    current_prices[ticker] = S
    
    # Compute daily log returns for volatility estimation
    log_returns = np.log(prices).diff().dropna()
    daily_var = log_returns.var().item()
    annual_var = daily_var * 252
    sigma = math.sqrt(annual_var)
    sigmas[ticker] = sigma
    
    # Set strike price as 90% of current price (seller's perspective)
    K = 0.9 * S
    strike_prices[ticker] = K
    
    # Compute Black-Scholes parameters for a put option
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Black-Scholes put option price (premium received by the seller)
    premium = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    put_prices[ticker] = premium
    
    # Also compute daily arithmetic returns (for simulation)
    returns_daily = prices.pct_change().dropna()
    daily_means[ticker] = returns_daily.mean().item()
    daily_variances[ticker] = returns_daily.var().item() # Edit to not use percentage change, use daily var instead
    
    print(f"----- {ticker} Put Option Pricing -----")
    print(f"Current Stock Price: {S:.2f}")
    print(f"Annualized Volatility: {sigma*100:.2f}%")
    print(f"Strike Price: {K:.2f}")
    print(f"Put Option Price (Premium): {premium:.2f}\n")

# ------------------------------
# Step 2: Vine Copula Modeling on Daily Returns (Cross-Sectional)
# ------------------------------

# Download merged price data for the tickers
dfs = []
for ticker in tickers:
    df = yf.download(ticker, period="1y", auto_adjust=True)[['Close']]
    df.rename(columns={'Close': ticker}, inplace=True)
    dfs.append(df)
prices_df = pd.concat(dfs, axis=1).dropna()

# Calculate arithmetic daily returns using pct_change()
returns_df = prices_df.pct_change().dropna()

# Transform each return series to uniform margins using an empirical CDF
def to_uniform(series):
    return (rankdata(series) - 0.5) / len(series)

uniform_data = returns_df.apply(to_uniform, axis=0)

# Fit the vine copula model on the uniform data (captures cross-sectional dependency)
vine_model = pvc.Vinecop.from_data(uniform_data.values)
print("\n----- Vine Copula Model Summary -----")
print(vine_model)

# ------------------------------
# Step 3: Simulate T-period (e.g., 126-day) Trajectories using Daily Vine Copula Simulation
# ------------------------------

num_simulations = 1000

# For each day, simulate num_simulations joint uniform observations.
# (Assume daily returns are independent across days.)
# This creates a list of arrays: each array has shape (num_simulations, num_stocks)
daily_uniform_list = []
for day in range(num_days):
    daily_uniform_list.append(vine_model.simulate(num_simulations))
# Convert list to array: shape (num_days, num_simulations, num_stocks)
daily_uniform = np.array(daily_uniform_list)
# Rearrange to shape (num_simulations, num_days, num_stocks)
daily_uniform = np.transpose(daily_uniform, (1, 0, 2))

# Print first 5 rows of simulated uniform data for inspection
print("\n----- Simulated Uniform Data (first 5 rows) -----")
print(daily_uniform[:5])

# Convert daily uniform values to daily simulated returns.
# For each stock, simulated return = mean + sqrt(variance) * norm.ppf(u)
# Shape of simulated_daily_returns: (num_simulations, num_days, num_stocks)
simulated_daily_returns = np.zeros_like(daily_uniform)
for j, ticker in enumerate(tickers):
    m = daily_means[ticker] #USE MEAN OF LOG RETURNS
    sigma_daily = math.sqrt(daily_variances[ticker]) #use daily VaR instead
    simulated_daily_returns[:, :, j] = m + sigma_daily * norm.ppf(daily_uniform[:, :, j])

# Print first 5 rows of simulated daily returns for AAPL
print("\n----- Simulated Daily Returns for AAPL (first 5 rows) -----")
sim_returns_AAPL = pd.DataFrame(simulated_daily_returns[:, :, 0])
print(sim_returns_AAPL.head())

# For each simulation and each stock, compound the daily returns over T days.
# cumulative_return = prod_{d=1}^{num_days} (1 + r_d) - 1
# This results in cumulative_returns with shape (num_simulations, num_stocks)
cumulative_returns = np.prod(1 + simulated_daily_returns, axis=1) - 1 # use summation of log returns instead of product. Un-log it

# Compute simulated future prices for each stock:
simulated_future_prices = {}
for j, ticker in enumerate(tickers):
    S = current_prices[ticker]
    simulated_future_prices[ticker] = S * (1 + cumulative_returns[:, j]) #instead of 1+, use un-log version

# ------------------------------
# Step 4: Loss Distribution & VaR Calculation for Short Puts on All Stocks
# ------------------------------

VaRs = {}
for ticker in tickers:
    S = current_prices[ticker]
    K = strike_prices[ticker]
    premium = put_prices[ticker]
    sim_prices = simulated_future_prices[ticker]
    
    # For a short put position (seller's perspective):
    # If simulated price < strike, loss = (Strike - Simulated Price) - Premium.
    # Otherwise, loss = -Premium (i.e. a gain of the premium).
    losses = np.where(sim_prices < K, (K - sim_prices) - premium, -premium)
    
    VaR = np.quantile(losses, 0.95)
    VaRs[ticker] = VaR
    
    print(f"\n----- Loss Distribution for Short Put on {ticker} -----")
    print(pd.Series(losses).describe())
    print(f"At 95% confidence, the VaR for {ticker} is: {VaR:.2f}")


# instead of calcing each VaR individually, calculate loss for portfolio for each iteration of simulation. Then, calculate VaR based on the losses observed in sims
# L - (1+i)^t(pA+pM+pG) per iteration

# Calculate total capital required (sum of individual VaRs)
total_capital_required = sum(VaRs[ticker] for ticker in tickers)
print("\n----- Summary for Seller -----")
print(f"Total capital required to cover worst-case losses: {total_capital_required:.2f}")

print("\nIdeal Premiums (based on 95% VaR) for selling the puts:")
for ticker in tickers:
    print(f"{ticker}: {VaRs[ticker]:.2f}")

# ------------------------------
# Visualization: Histograms
# ------------------------------

# 1. Histogram of Simulated Future Prices for each stock
for ticker in tickers:
    plt.figure()
    sim_prices = simulated_future_prices[ticker]
    plt.hist(sim_prices, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=strike_prices[ticker], color='red', linestyle='--', label='Strike Price')
    plt.title(f"Histogram of Simulated Future Prices for {ticker}")
    plt.xlabel("Simulated Future Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# 2. Histogram of Loss Distributions for each stock
for ticker in tickers:
    plt.figure()
    sim_prices = simulated_future_prices[ticker]
    premium = put_prices[ticker]
    # Compute losses again for plotting clarity:
    losses = np.where(sim_prices < strike_prices[ticker], 
                      (strike_prices[ticker] - sim_prices) - premium, -premium)
    plt.hist(losses, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=np.quantile(losses, 0.95), color='red', linestyle='--', label='95th Percentile VaR')
    plt.title(f"Histogram of Loss Distribution for Short Put on {ticker}")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# VaR of combined porfolio over 1000 sims
# Use data from calendar year 2024