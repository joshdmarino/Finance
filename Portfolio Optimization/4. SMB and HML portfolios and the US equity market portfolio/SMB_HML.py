#!/usr/bin/env python3
"""
Portfolio Strategies for SMB, HML, and US Equity Market Data
----------------------------------------------------------------
This script reads Fama–French factor data and SPY (US equity market) returns,
converts the factor values from percentages to decimals, computes excess returns
(by subtracting the risk-free rate), and then applies 15 portfolio allocation strategies.
Performance is measured on daily returns and includes:
    - Cumulative Return
    - Annualized Return (252 trading days)
    - Annualized Volatility
    - Sharpe Ratio
    - Certainty–Equivalent Return (CEQ)
    - Turnover (set to NaN since weights are static)
    
IMPORTANT: The Fama–French CSV file contains extra header rows and factor values in percentages.
We convert the values by dividing by 100.
    
The 15 strategies implemented include:
  1. Naïve Diversification (1/N)
  2. Sample–Based Mean–Variance (MV) – Closed–form (unconstrained)
  3. Bayesian Diffuse–Prior Portfolio
  4. Bayes–Stein Portfolio
  5. Bayesian Data–and–Model (DM) Portfolio
  6. Minimum–Variance (Min) Portfolio
  7. Value–Weighted Market Portfolio (VW)
  8. MacKinlay and Pastor’s Missing–Factor Model (MP)
  9. MV with Shortsale Constraints (MV-C)
 10. Bayes–Stein with Shortsale Constraints (BS-C)
 11. Minimum–Variance with Shortsale Constraints (Min-C)
 12. Minimum–Variance with Generalized Constraints (G-Min-C)
 13. Kan and Zhou’s Three–Fund Model (KZ)
 14. Mixture of Minimum–Variance and 1/N (EW-Min)
 15. Garlappi, Uppal, and Wang’s (GUW) Model

The final performance results are printed and exported to a CSV file.
"""

# =============================================================================
# Section 1: Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# =============================================================================
# Section 2: Data Loading and Preprocessing
# =============================================================================
def get_fama_french_data(file_path):
    """
    Load Fama–French factor data (Mkt-RF, SMB, HML, RF) from a CSV file.
    The file contains extra header rows (skiprows=5) and factor values in percentages.
    We convert these values into decimals by dividing by 100.
    
    Returns:
        DataFrame with Date (as datetime index) and columns: "Mkt-RF", "SMB", "HML", "RF".
    """
    ff_data = pd.read_csv(file_path, skiprows=5, header=None,
                          names=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    # Filter to keep rows where Date is an 8-digit string
    ff_data = ff_data[ff_data["Date"].astype(str).str.match(r"^\d{8}$")]
    ff_data["Date"] = pd.to_datetime(ff_data["Date"], format="%Y%m%d", errors="coerce")
    ff_data = ff_data.dropna(subset=["Date"])
    # Convert factor values from percentages to decimals
    ff_data[["Mkt-RF", "SMB", "HML", "RF"]] = ff_data[["Mkt-RF", "SMB", "HML", "RF"]].astype(float) / 100.0
    return ff_data.set_index("Date")

def get_sp_sector_data(tickers, start_date, end_date):
    """
    Download daily adjusted close prices for the given tickers from Yahoo Finance,
    then compute daily percentage returns.
    
    Parameters:
        tickers (list): List of ticker symbols (e.g. ["SPY"]).
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    
    Returns:
        DataFrame of daily returns.
    """
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    returns = data.pct_change()
    return returns

# =============================================================================
# Section 3: Compute Excess Returns and Sample Statistics
# =============================================================================
# Load Fama–French data (for factors and risk–free rate)
ff_file_path = "Portfolio Optimization/4. SMB and HML portfolios and the US equity market portfolio/F-F_Research_Data_Factors_daily.CSV"
fama_french_data = get_fama_french_data(ff_file_path)

# Load SPY returns (representing the US equity market)
sp_tickers = ["SPY"]
sector_returns = get_sp_sector_data(sp_tickers, "2003-01-01", "2024-12-31")

# Merge SPY returns with Fama–French data (using dates as key)
merged_data = pd.merge(sector_returns, fama_french_data, left_index=True, right_index=True, how="inner")
# The merged DataFrame columns are expected to be: ["SPY", "Mkt-RF", "SMB", "HML", "RF"]

# Compute excess returns: subtract the risk-free rate from each asset return.
# We drop the "RF" column first, then subtract the "RF" series.
excess_returns = merged_data.drop(columns=["RF"]).sub(merged_data["RF"], axis=0)
# Now excess_returns should have 4 columns: e.g., ["SPY", "Mkt-RF", "SMB", "HML"]

# Compute sample statistics (mean and covariance) from the excess returns.
mean_returns = excess_returns.mean()
cov_matrix = excess_returns.cov()

# Align the risk–free rate series with the excess_returns index.
rf_series = fama_french_data["RF"].reindex(excess_returns.index, method="ffill")
rf_avg = rf_series.mean()  # average daily risk–free rate

# =============================================================================
# Section 4: Performance Metrics Functions
# =============================================================================
def compute_performance(portfolio_returns, rf_series):
    """
    Compute performance metrics from daily portfolio returns.
    
    Metrics:
      - Cumulative Return: (1 + returns product) - 1
      - Annualized Return: Compounded over 252 trading days.
      - Annualized Volatility: Standard deviation scaled by sqrt(252)
      - Sharpe Ratio: (Annualized Return - mean RF) / Annualized Volatility
    """
    port_returns = portfolio_returns.dropna()
    cumulative_return = (1 + port_returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(port_returns)) - 1
    annualized_vol = port_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - rf_series.mean()) / annualized_vol
    return cumulative_return, annualized_return, annualized_vol, sharpe

def compute_CEQ(annualized_return, annualized_vol, gamma):
    """
    Compute Certainty–Equivalent Return (CEQ) for a mean–variance investor.
    
    Formula:
      CEQ = Annualized Return - (gamma/2) * (Annualized Volatility)^2
    """
    return annualized_return - (gamma / 2) * (annualized_vol ** 2)

# =============================================================================
# Section 5: Portfolio Optimization Functions (15 Strategies)
# =============================================================================
# (See previous explanations; each function is similar to earlier versions.)

def equal_weight_portfolio(n_assets):
    """Return an equal–weight portfolio (each asset gets 1/n_assets)."""
    return np.ones(n_assets) / n_assets

def mean_variance_closed_form(mean_returns, cov_matrix):
    """
    Compute the unconstrained mean–variance portfolio (closed–form).
    Formula: w = inv(cov) * mean_returns / (1^T * inv(cov) * mean_returns)
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_returns))
    num = inv_cov.dot(mean_returns)
    denom = ones.T.dot(num)
    weights = num / denom
    return weights

def bayesian_diffuse_prior(mean_returns, T):
    """Compute the diffuse–prior adjusted mean: (T/(T+1))*sample_mean."""
    return (T / (T + 1)) * mean_returns

def bayes_stein_portfolio(mean_returns, cov_matrix):
    """Compute the Bayes–Stein shrunk mean estimate."""
    n = len(mean_returns)
    global_mean = mean_returns.mean()
    total_variance = np.trace(cov_matrix) / n
    sample_variance = mean_returns.var()
    phi = sample_variance / (sample_variance + total_variance)
    return global_mean + phi * (mean_returns - global_mean)

def bayesian_data_and_model_portfolio(mean_returns, T):
    """Compute the DM adjusted mean: (T/(T+1))*sample_mean + (1/(T+1))*global_mean."""
    global_mean = mean_returns.mean()
    lambda_ = T / (T + 1)
    return lambda_ * mean_returns + (1 - lambda_) * global_mean

def minimum_variance_portfolio(cov_matrix):
    """Compute the minimum–variance portfolio (ignoring expected returns)."""
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    return inv_cov.dot(ones) / (ones.T.dot(inv_cov.dot(ones)))

def value_weighted_market_portfolio(tickers):
    """
    If 'SPY' is among the tickers (assumed to represent the market),
    assign 100% weight to SPY; otherwise, use equal weighting.
    IMPORTANT: Pass the list of asset names from excess_returns (which excludes RF).
    """
    if "SPY" in tickers:
        weights = np.zeros(len(tickers))
        weights[tickers.index("SPY")] = 1.0
    else:
        weights = equal_weight_portfolio(len(tickers))
    return weights

def mackinlay_pastor_portfolio(mean_returns, cov_matrix):
    """A placeholder: return the MV closed–form weights."""
    return mean_variance_closed_form(mean_returns, cov_matrix)

def mean_variance_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    Compute the mean–variance portfolio with no short–selling (weights in [0,1]).
    Uses numerical optimization (SLSQP) to maximize the Sharpe ratio.
    """
    n = len(mean_returns)
    def objective(w):
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -((port_return - risk_free_rate) / port_vol)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def bayes_stein_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """Apply Bayes–Stein shrinkage then solve for the constrained MV portfolio."""
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    return mean_variance_optimization_constrained(bs_mean, cov_matrix, risk_free_rate)

def minimum_variance_constrained(cov_matrix):
    """
    Compute the minimum–variance portfolio with no short–selling constraints.
    Uses numerical optimization (SLSQP) with weights in [0,1].
    """
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def minimum_variance_generalized(cov_matrix, floor=0.05):
    """
    Compute the minimum–variance portfolio with an additional floor constraint on each weight.
    Each weight must be at least 'floor'.
    """
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] - floor})
    bounds = [(None, None)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def kan_zhou_three_fund(mean_returns, cov_matrix, risk_free_rate, gamma=3.0, hedging_coef=-0.1):
    """
    Implement Kan and Zhou’s stylized three–fund model.
      1. Myopic component: inv(cov) * (mean_returns - risk_free_rate)
      2. Hedging component: inv(cov) * (mean_returns - BS_shrunk_mean)
      3. Combined weights: (1/gamma)*myopic + hedging_coef*hedging
      4. Normalize by dividing by the sum of absolute values.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    myopic = inv_cov.dot(mean_returns - risk_free_rate)
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    hedging = inv_cov.dot(mean_returns - bs_mean)
    weights = (1 / gamma) * myopic + hedging_coef * hedging
    weights = weights / np.sum(np.abs(weights))
    return weights

def mixture_minvar_equal(cov_matrix, lambda_mix=0.5):
    """
    Compute a convex combination of the minimum–variance portfolio and the equal–weight portfolio.
    lambda_mix is the weight for the minimum–variance portfolio.
    """
    w_min = minimum_variance_portfolio(cov_matrix)
    w_equal = equal_weight_portfolio(len(cov_matrix))
    return lambda_mix * w_min + (1 - lambda_mix) * w_equal

def garlappi_uppal_wang(mean_returns, cov_matrix, risk_free_rate, tau=0.1):
    """
    Implement a simplified version of the GUW model.
    Adjust the sample mean by shrinking it toward the risk–free rate:
         adjusted_mean = (1 - tau)*sample_mean + tau*risk_free_rate
    Then compute the MV closed–form portfolio.
    """
    adjusted_mean = (1 - tau) * mean_returns + tau * risk_free_rate
    return mean_variance_closed_form(adjusted_mean, cov_matrix)

# =============================================================================
# Section 6: Main Execution – Compute Portfolio Weights and Performance
# =============================================================================
if __name__ == "__main__":
    # Data Input Parameters
    ff_file_path = "Portfolio Optimization/4. SMB and HML portfolios and the US equity market portfolio/F-F_Research_Data_Factors_daily.CSV"
    
    # Load Fama–French data and SPY data
    fama_french_data = get_fama_french_data(ff_file_path)
    sp_tickers = ["SPY"]  # Using SPY to represent the US equity market
    sector_returns = get_sp_sector_data(sp_tickers, "2003-01-01", "2024-12-31")
    
    # Merge SPY returns with Fama–French data on date (inner join)
    merged_data = pd.merge(sector_returns, fama_french_data, left_index=True, right_index=True, how="inner")
    # merged_data columns: e.g., ["SPY", "Mkt-RF", "SMB", "HML", "RF"]
    
    # Calculate Excess Returns: subtract the risk–free rate from the asset returns.
    # We drop the "RF" column and subtract the "RF" series from each asset return.
    excess_returns = merged_data.drop(columns=["RF"]).sub(merged_data["RF"], axis=0)
    # Now excess_returns should have 4 columns (e.g., ["SPY", "Mkt-RF", "SMB", "HML"])
    
    # Compute sample statistics from excess returns.
    mean_returns = excess_returns.mean()
    cov_matrix = excess_returns.cov()
    
    # Set performance parameters.
    n_assets = len(excess_returns.columns)  # should be 4
    T = len(excess_returns)                 # number of trading days
    rf_series = fama_french_data["RF"].reindex(excess_returns.index, method="ffill")
    rf_avg = rf_series.mean()               # average daily risk–free rate
    gamma_risk = 3.0                        # risk–aversion parameter for CEQ
    
    # Compute portfolio weights for each strategy.
    strategy_weights = {}
    
    # Strategy 1: Naïve Diversification (1/N)
    strategy_weights["1/N"] = equal_weight_portfolio(n_assets)
    
    # Strategy 2: Sample–Based Mean–Variance (MV)
    strategy_weights["MV"] = mean_variance_closed_form(mean_returns, cov_matrix)
    
    # Strategy 3: Bayesian Diffuse–Prior Portfolio
    diffused_mean = bayesian_diffuse_prior(mean_returns, T)
    strategy_weights["Bayesian Diffuse-Prior"] = mean_variance_closed_form(diffused_mean, cov_matrix)
    
    # Strategy 4: Bayes–Stein Portfolio
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    strategy_weights["Bayes-Stein"] = mean_variance_closed_form(bs_mean, cov_matrix)
    
    # Strategy 5: Bayesian Data–and–Model (DM) Portfolio
    dm_mean = bayesian_data_and_model_portfolio(mean_returns, T)
    strategy_weights["Bayesian Data-and-Model"] = mean_variance_closed_form(dm_mean, cov_matrix)
    
    # Strategy 6: Minimum–Variance (Min) Portfolio
    strategy_weights["Min"] = minimum_variance_portfolio(cov_matrix)
    
    # Strategy 7: Value–Weighted Market Portfolio (VW)
    # IMPORTANT: Use the column names from excess_returns (not merged_data) so that we have 4 assets.
    strategy_weights["VW"] = value_weighted_market_portfolio(list(excess_returns.columns))
    
    # Strategy 8: MacKinlay and Pastor’s Missing–Factor Model (MP)
    strategy_weights["MP"] = mackinlay_pastor_portfolio(mean_returns, cov_matrix)
    
    # Strategy 9: MV with Shortsale Constraints (MV-C)
    strategy_weights["MV-C"] = mean_variance_optimization_constrained(mean_returns, cov_matrix, rf_avg)
    
    # Strategy 10: Bayes–Stein with Shortsale Constraints (BS-C)
    strategy_weights["BS-C"] = bayes_stein_optimization_constrained(mean_returns, cov_matrix, rf_avg)
    
    # Strategy 11: Minimum–Variance with Shortsale Constraints (Min-C)
    strategy_weights["Min-C"] = minimum_variance_constrained(cov_matrix)
    
    # Strategy 12: Minimum–Variance with Generalized Constraints (G-Min-C)
    strategy_weights["G-Min-C"] = minimum_variance_generalized(cov_matrix, floor=0.05)
    
    # Strategy 13: Kan and Zhou’s Three–Fund Model (KZ)
    strategy_weights["KZ"] = kan_zhou_three_fund(mean_returns, cov_matrix, rf_avg, gamma=3.0, hedging_coef=-0.1)
    
    # Strategy 14: Mixture of Minimum–Variance and 1/N (EW-Min)
    strategy_weights["EW-Min"] = mixture_minvar_equal(cov_matrix, lambda_mix=0.5)
    
    # Strategy 15: Garlappi, Uppal, and Wang’s (GUW) Model
    strategy_weights["GUW"] = garlappi_uppal_wang(mean_returns, cov_matrix, rf_avg, tau=0.1)
    
    # Compute portfolio returns and performance metrics for each strategy.
    performance_results = {}
    
    for strat_name, weights in strategy_weights.items():
        # Calculate daily portfolio returns as the weighted sum of excess returns.
        port_returns = (excess_returns * weights).sum(axis=1)
        cum_ret, ann_ret, ann_vol, sharpe = compute_performance(port_returns.dropna(), rf_series)
        ceq = compute_CEQ(ann_ret, ann_vol, gamma_risk)
        performance_results[strat_name] = {
            "Cumulative Return": cum_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "CEQ": ceq,
            "Turnover": np.nan
        }
    
    # Aggregate the results into a DataFrame and export to CSV.
    results_df = pd.DataFrame(performance_results).T
    results_df.index.name = "Strategy"
    results_df = results_df.reset_index()
    
    print("\nSMB, HML, and US Equity Market Portfolio Strategy Performance:")
    print(results_df)
    
    results_df.to_csv("Portfolio Optimization/4. SMB and HML portfolios and the US equity market portfolio/portfolio_smb_hml_results.csv", index=False)
    print("\nResults exported to 'portfolio_smb_hml_results.csv'")
