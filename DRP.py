import yfinance as yf
import numpy as np
import math
import pandas as pd
from scipy.stats import norm, rankdata
import pyvinecopulib as pvc
import matplotlib.pyplot as plt

# ------------------------------
# Set Date Range for Historical Data
# ------------------------------
start_date = "2024-01-01"
end_date   = "2024-12-31"

# ------------------------------
# Step 1: Price Puts using Black-Scholes for Each Stock
# ------------------------------
tickers = ['AAPL', 'MSFT', 'GOOG']
T = 0.5  # Time to expiration in years
num_days = int(T * 252)  # Approximate number of trading days in T years

# Retrieve the risk-free rate from the 10-year treasury yield ("^TNX")
tnx = yf.Ticker("^TNX")
tnx_data = tnx.history(period="1d", start=start_date, end=end_date)
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
# For simulation, compute daily log return means and variances
daily_log_means = {}
daily_log_variances = {}

for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
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
    
    # For simulation, compute the mean and variance of daily log returns
    daily_log_means[ticker] = log_returns.mean().item()
    daily_log_variances[ticker] = log_returns.var().item()
    
    print(f"----- {ticker} Put Option Pricing -----")
    print(f"Current Stock Price: {S:.2f}")
    print(f"Annualized Volatility: {sigma*100:.2f}%")
    print(f"Strike Price: {K:.2f}")
    print(f"Put Option Price (Premium): {premium:.2f}\n")

# ------------------------------
# Step 2: Vine Copula Modeling on Daily Log Returns (Cross-Sectional)
# ------------------------------
dfs = []
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)[['Close']]
    df.rename(columns={'Close': ticker}, inplace=True)
    dfs.append(df)
prices_df = pd.concat(dfs, axis=1).dropna()

# Calculate daily log returns for the merged data
log_returns_df = np.log(prices_df).diff().dropna()

# Transform each log return series to uniform margins using an empirical CDF
def to_uniform(series):
    return (rankdata(series) - 0.5) / len(series)

uniform_data = log_returns_df.apply(to_uniform, axis=0)

# Fit the vine copula model on the uniform data (captures cross-sectional dependency)
vine_model = pvc.Vinecop.from_data(uniform_data.values)
print("\n----- Vine Copula Model Summary -----")
print(vine_model)

# ------------------------------
# Step 3: Simulate T-period (e.g., 126-day) Trajectories using Daily Vine Copula Simulation
# ------------------------------
num_simulations = 1000
daily_uniform_list = []
for day in range(num_days):
    daily_uniform_list.append(vine_model.simulate(num_simulations))
daily_uniform = np.array(daily_uniform_list)
daily_uniform = np.transpose(daily_uniform, (1, 0, 2))

print("\n----- Simulated Uniform Data (first 5 rows) -----")
print(daily_uniform[:5])

# Convert to simulated daily log returns:
simulated_daily_log_returns = np.zeros_like(daily_uniform)
for j, ticker in enumerate(tickers):
    m = daily_log_means[ticker]
    sigma_daily = math.sqrt(daily_log_variances[ticker])
    simulated_daily_log_returns[:, :, j] = m + sigma_daily * norm.ppf(daily_uniform[:, :, j])

print("\n----- Simulated Daily Log Returns for AAPL (first 5 rows) -----")
sim_returns_AAPL = pd.DataFrame(simulated_daily_log_returns[:, :, 0])
print(sim_returns_AAPL.head())

cumulative_log_returns = np.sum(simulated_daily_log_returns, axis=1)
simulated_future_prices = {}
for j, ticker in enumerate(tickers):
    S = current_prices[ticker]
    simulated_future_prices[ticker] = S * np.exp(cumulative_log_returns[:, j])

# ------------------------------
# Step 4: Loss Distribution & Portfolio VaR Calculation for Short Puts (Seller's Perspective)
# ------------------------------
portfolio_losses = np.zeros(num_simulations)
for ticker in tickers:
    S = current_prices[ticker]
    K = strike_prices[ticker]
    premium = put_prices[ticker]
    sim_prices = simulated_future_prices[ticker]
    portfolio_losses += np.maximum(K - sim_prices, 0)
    
#compounded_total_premium = (1 + r)**T * sum(put_prices[ticker] for ticker in tickers)
portfolio_losses = portfolio_losses 
portfolio_VaR = np.quantile(portfolio_losses, 0.95)

print("\n----- Portfolio Loss Distribution -----")
print(pd.Series(portfolio_losses).describe())
print(f"\nAt 95% confidence, the portfolio VaR is: {portfolio_VaR:.2f}")

# ------------------------------
# (Optional) Calculate individual stock VaRs for reference
# ------------------------------
individual_VaRs = {}
for ticker in tickers:
    S = current_prices[ticker]
    K = strike_prices[ticker]
    premium = put_prices[ticker]
    sim_prices = simulated_future_prices[ticker]
    losses = np.where(sim_prices < K, (K - sim_prices) - premium, -premium)
    individual_VaRs[ticker] = np.quantile(losses, 0.95)
    print(f"\n----- Loss Distribution for Short Put on {ticker} -----")
    print(pd.Series(losses).describe())
    print(f"At 95% confidence, the VaR for {ticker} is: {individual_VaRs[ticker]:.2f}")

print("\n----- Summary for Seller -----")
print(f"Total portfolio VaR (capital required to cover worst-case losses): {portfolio_VaR:.2f}")
print("\nIdeal Premiums (based on individual 95% VaR) for selling the puts:")
for ticker in tickers:
    print(f"{ticker}: {individual_VaRs[ticker]:.2f}")

# ------------------------------
# Visualization: Histograms of Loss Distributions Only
# ------------------------------
for ticker in tickers:
    plt.figure()
    sim_prices = simulated_future_prices[ticker]
    premium = put_prices[ticker]
    losses = np.where(sim_prices < strike_prices[ticker],
                      (strike_prices[ticker] - sim_prices) - premium, -premium)
    plt.hist(losses, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=np.quantile(losses, 0.95), color='red', linestyle='--', label='95th Percentile VaR')
    plt.title(f"Histogram of Loss Distribution for Short Put on {ticker}")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    # Uncomment the next line to save the figure to a file instead of displaying
    # plt.savefig(f"{ticker}_loss_histogram.png")
    plt.show()
    
# Optionally, plot portfolio loss histogram:
plt.figure()
plt.hist(portfolio_losses, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(x=portfolio_VaR, color='red', linestyle='--', label='95th Percentile Portfolio VaR')
plt.title("Histogram of Portfolio Loss Distribution")
plt.xlabel("Portfolio Loss")
plt.ylabel("Frequency")
plt.legend()
# Uncomment the next line to save the figure to a file instead of displaying
# plt.savefig("portfolio_loss_histogram.png")
plt.show()
