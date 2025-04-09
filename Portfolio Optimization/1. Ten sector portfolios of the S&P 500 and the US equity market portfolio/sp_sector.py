#!/usr/bin/env python3
"""
Portfolio Strategies Comparison Script with CEQ Metric
-------------------------------------------------------
This script implements 15 portfolio allocation strategies (including the naïve 1/N benchmark
and simple mean-variance) and computes performance metrics. In addition to the standard metrics,
we compute the Certainty-Equivalent Return (CEQ) for each strategy using a specified risk-aversion
parameter (γ).

Note: Turnover (measuring trading frequency) requires dynamic rebalancing data. Because this script
uses a single static allocation (full-sample optimization), turnover is not computed here.

The performance metrics include:
  - Cumulative Return
  - Annualized Return
  - Annualized Volatility
  - Sharpe Ratio
  - Certainty-Equivalent Return (CEQ)

You can export the results to a CSV file for further analysis in Excel.
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
#
# Change the following variables to use different data.
#   - sp_tickers: list of asset tickers (e.g., S&P sector ETFs + SPY)
#   - sp_start_date, sp_end_date: the date range for the historical data
#   - ff_file_path: file path to the Fama-French CSV for risk-free rates
# =============================================================================
sp_tickers = ["XLF", "XLK", "XLY", "XLI", "XLE", "XLV", "XLB", "XLU", "XLRE", "XLC", "SPY"]
sp_start_date = "2003-01-01"
sp_end_date   = "2024-12-31"
ff_file_path  = "Portfolio Optimization/1. Ten sector portfolios of the S&P 500 and the US equity market portfolio/F-F_Research_Data_Factors_daily.CSV"

def get_rf_data(file_path):
    """
    Load risk-free rate data from the Fama-French CSV.
    Assumes the CSV has rows starting at row 5 with columns:
      Date, Mkt-RF, SMB, HML, RF
    """
    ff_data = pd.read_csv(file_path, skiprows=5, header=None,
                          names=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    # Filter to rows where Date is an 8-digit string.
    ff_data = ff_data[ff_data["Date"].astype(str).str.match(r"^\d{8}$")]
    ff_data["Date"] = pd.to_datetime(ff_data["Date"], format="%Y%m%d", errors="coerce")
    ff_data = ff_data.dropna(subset=["Date"])
    return ff_data[["Date", "RF"]]

def get_sp_sector_data(tickers, start_date, end_date):
    """
    Download adjusted close prices for the tickers and compute daily returns.
    """
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    returns = data.pct_change()
    return returns

# Load the asset returns and risk-free data
sector_returns = get_sp_sector_data(sp_tickers, sp_start_date, sp_end_date)
rf_data = get_rf_data(ff_file_path)

# Align datetime indices and merge data
sector_returns.index = pd.to_datetime(sector_returns.index).tz_localize(None)
rf_data["Date"] = pd.to_datetime(rf_data["Date"], errors='coerce')
merged_data = pd.merge(sector_returns, rf_data, left_index=True, right_on="Date", how="left")
merged_data.set_index("Date", inplace=True)

# Compute excess returns: asset returns minus the risk-free rate
excess_returns = merged_data.drop(columns=["RF"]).sub(merged_data["RF"], axis=0)
numeric_data = excess_returns.dropna()  # remove missing rows

# Compute sample statistics using the excess returns data:
mean_returns = numeric_data.mean()        # sample mean vector (excess returns)
cov_matrix   = numeric_data.cov()           # sample covariance matrix

# Align the risk-free series with the asset returns
rf_series = rf_data.set_index("Date")["RF"].reindex(sector_returns.index, method="ffill")

# =============================================================================
# Section 3: Performance Metrics Calculation
# =============================================================================
def compute_performance(portfolio_returns, rf_series):
    """
    Compute key performance metrics:
      - Cumulative Return over the sample period
      - Annualized Return (assuming 252 trading days)
      - Annualized Volatility
      - Sharpe Ratio (using the mean risk-free rate)
    """
    port_returns = portfolio_returns.dropna()
    cumulative_return = (1 + port_returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(port_returns)) - 1
    annualized_vol = port_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - rf_series.mean()) / annualized_vol
    return cumulative_return, annualized_return, annualized_vol, sharpe

def compute_CEQ(annualized_return, annualized_vol, gamma):
    """
    Compute the Certainty-Equivalent Return (CEQ) for a mean-variance investor.
    Formula: CEQ = Annualized Return - (γ/2) * (Annualized Volatility)^2
    gamma: risk-aversion parameter.
    """
    return annualized_return - (gamma / 2) * (annualized_vol ** 2)

# =============================================================================
# Section 4: Portfolio Optimization Functions
#
# Each function corresponds to one of the 15 strategies.
# Detailed comments explain the formulas and constraints.
# =============================================================================

# Strategy 1: Naïve Diversification (1/N)
def equal_weight_portfolio(n_assets):
    """Return an equal-weight portfolio for n_assets (each weight = 1/n_assets)."""
    return np.ones(n_assets) / n_assets

# Strategy 2: Simple Sample-Based Mean-Variance (MV) using closed-form solution
def mean_variance_closed_form(mean_returns, cov_matrix):
    """
    Compute the unconstrained mean-variance portfolio (closed-form solution).
    Formula: w_MV = inv(cov) * mean_returns / (1^T * inv(cov) * mean_returns)
    Allows short-selling.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_returns))
    num = inv_cov.dot(mean_returns)
    denom = ones.T.dot(num)
    weights = num / denom
    return weights

# Strategy 3: Bayesian Diffuse-Prior Portfolio
def bayesian_diffuse_prior(mean_returns, T):
    """
    Adjust sample mean using a diffuse (flat) prior.
    diffused_mean = (T/(T+1)) * sample_mean, where T is the sample size.
    """
    return (T / (T + 1)) * mean_returns

# Strategy 4: Bayes–Stein Portfolio
def bayes_stein_portfolio(mean_returns, cov_matrix):
    """
    Compute Bayes–Stein shrinkage estimates.
    Shrinks the sample mean toward the global (grand) mean.
    """
    n = len(mean_returns)
    global_mean = mean_returns.mean()
    total_variance = np.trace(cov_matrix) / n
    sample_variance = mean_returns.var()
    phi = sample_variance / (sample_variance + total_variance)
    bs_mean = global_mean + phi * (mean_returns - global_mean)
    return bs_mean

# Strategy 5: Bayesian Data-and-Model (DM) Portfolio
def bayesian_data_and_model_portfolio(mean_returns, T):
    """
    Combine the sample mean with the global mean using Bayesian weights.
    DM_mean = (T/(T+1)) * sample_mean + (1/(T+1)) * global_mean.
    """
    global_mean = mean_returns.mean()
    lambda_ = T / (T + 1)
    dm_mean = lambda_ * mean_returns + (1 - lambda_) * global_mean
    return dm_mean

# Strategy 6: Minimum-Variance (Min) Portfolio (closed-form)
def minimum_variance_portfolio(cov_matrix):
    """
    Compute the minimum-variance portfolio that does not rely on expected returns.
    Formula: w_min = inv(cov) * ones / (ones^T * inv(cov) * ones)
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    weights = inv_cov.dot(ones) / (ones.T.dot(inv_cov.dot(ones)))
    return weights

# Strategy 7: Value-Weighted Market Portfolio (VW)
def value_weighted_market_portfolio(tickers):
    """
    If 'SPY' is in the ticker list (assumed to represent the market portfolio),
    assign all weight to 'SPY'; otherwise, default to equal weights.
    """
    if "SPY" in tickers:
        weights = np.zeros(len(tickers))
        weights[tickers.index("SPY")] = 1.0
    else:
        weights = equal_weight_portfolio(len(tickers))
    return weights

# Strategy 8: MacKinlay and Pastor’s Missing-Factor Model (MP)
def mackinlay_pastor_portfolio(mean_returns, cov_matrix):
    """
    Placeholder for MacKinlay and Pastor's missing-factor model.
    For demonstration, we return the simple MV closed-form weights.
    """
    return mean_variance_closed_form(mean_returns, cov_matrix)

# Strategy 9: MV with Shortsale Constraints (MV-C)
def mean_variance_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    Solve the mean-variance optimization with no short-selling (weights in [0,1]).
    Objective: maximize Sharpe Ratio = (w^T mean_returns - r_f) / portfolio_volatility.
    Uses numerical optimization (SLSQP).
    """
    n = len(mean_returns)
    def objective(w):
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(w.T.dot(cov_matrix).dot(w))
        return -((port_return - risk_free_rate) / port_vol)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 10: Bayes–Stein with Shortsale Constraints (BS-C)
def bayes_stein_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    Apply Bayes–Stein shrinkage to mean returns, then solve for the MV portfolio with
    no short-selling constraints.
    """
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    weights = mean_variance_optimization_constrained(bs_mean, cov_matrix, risk_free_rate)
    return weights

# Strategy 11: Minimum-Variance with Shortsale Constraints (Min-C)
def minimum_variance_constrained(cov_matrix):
    """
    Compute the minimum-variance portfolio subject to no short-selling.
    Uses numerical optimization with weights in [0,1].
    """
    n = len(cov_matrix)
    def objective(w):
        return w.T.dot(cov_matrix).dot(w)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 12: Minimum-Variance with Generalized Constraints (G-Min-C)
def minimum_variance_generalized(cov_matrix, floor=0.05):
    """
    Compute the minimum-variance portfolio with additional floor constraints (each weight >= floor).
    Uses numerical optimization.
    """
    n = len(cov_matrix)
    def objective(w):
        return w.T.dot(cov_matrix).dot(w)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] - floor})
    bounds = [(None, None)] * n  # No explicit upper bound (or you can set it to 1)
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 13: Kan and Zhou’s Three-Fund Model (KZ)
def kan_zhou_three_fund(mean_returns, cov_matrix, risk_free_rate, gamma=3.0, hedging_coef=-0.1):
    """
    Implement a stylized three-fund model.
    1. Compute the myopic component: inv(cov) * (mean_returns - risk_free_rate)
    2. Compute the hedging component: inv(cov) * (mean_returns - BS_shrunk_mean)
    3. Combine: weights = (1/gamma) * myopic + hedging_coef * hedging, then normalize.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    myopic = inv_cov.dot(mean_returns - risk_free_rate)
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    hedging = inv_cov.dot(mean_returns - bs_mean)
    weights = (1 / gamma) * myopic + hedging_coef * hedging
    weights = weights / np.sum(np.abs(weights))
    return weights

# Strategy 14: Mixture of Minimum-Variance and 1/N (EW-Min)
def mixture_minvar_equal(cov_matrix, lambda_mix=0.5):
    """
    Compute a convex combination of the minimum-variance portfolio and the equal-weight portfolio.
    lambda_mix is the weight given to the minimum-variance portfolio.
    """
    w_min = minimum_variance_portfolio(cov_matrix)
    w_equal = equal_weight_portfolio(len(cov_matrix))
    weights = lambda_mix * w_min + (1 - lambda_mix) * w_equal
    return weights

# Strategy 15: Garlappi, Uppal, and Wang’s (GUW) Model
def garlappi_uppal_wang(mean_returns, cov_matrix, risk_free_rate, tau=0.1):
    """
    Simplified implementation of the GUW model.
    Adjust the sample mean by shrinking it toward the risk-free rate:
         adjusted_mean = (1-tau)*sample_mean + tau*risk_free_rate,
    then compute the MV closed-form portfolio.
    """
    adjusted_mean = (1 - tau) * mean_returns + tau * risk_free_rate
    weights = mean_variance_closed_form(adjusted_mean, cov_matrix)
    return weights

# =============================================================================
# Section 5: Main Execution - Compute Weights, Portfolio Returns, and Performance
# =============================================================================
if __name__ == "__main__":
    # Define the number of assets and sample size (T)
    n_assets = len(mean_returns)
    T = len(numeric_data)  # sample size for Bayesian adjustments
    rf_avg = rf_series.mean()  # average risk-free rate over the sample
    
    # Set risk aversion parameter gamma for CEQ calculation
    gamma_risk = 3.0

    # Dictionary to store computed weights for each strategy
    strategy_weights = {}

    # Strategy 1: Naïve Diversification (1/N)
    strategy_weights["1/N"] = equal_weight_portfolio(n_assets)

    # Strategy 2: Simple Sample-Based Mean-Variance (MV)
    strategy_weights["MV"] = mean_variance_closed_form(mean_returns, cov_matrix)

    # Strategy 3: Bayesian Diffuse-Prior Portfolio
    diffused_mean = bayesian_diffuse_prior(mean_returns, T)
    strategy_weights["Bayesian Diffuse-Prior"] = mean_variance_closed_form(diffused_mean, cov_matrix)

    # Strategy 4: Bayes–Stein Portfolio
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    strategy_weights["Bayes-Stein"] = mean_variance_closed_form(bs_mean, cov_matrix)

    # Strategy 5: Bayesian Data-and-Model (DM) Portfolio
    dm_mean = bayesian_data_and_model_portfolio(mean_returns, T)
    strategy_weights["Bayesian Data-and-Model"] = mean_variance_closed_form(dm_mean, cov_matrix)

    # Strategy 6: Minimum-Variance (Min) Portfolio
    strategy_weights["Min"] = minimum_variance_portfolio(cov_matrix)

    # Strategy 7: Value-Weighted Market Portfolio (VW)
    strategy_weights["VW"] = value_weighted_market_portfolio(sp_tickers)

    # Strategy 8: MacKinlay and Pastor’s Missing-Factor Model (MP)
    strategy_weights["MP"] = mackinlay_pastor_portfolio(mean_returns, cov_matrix)

    # Strategy 9: MV with Shortsale Constraints (MV-C)
    strategy_weights["MV-C"] = mean_variance_optimization_constrained(mean_returns, cov_matrix, rf_avg)

    # Strategy 10: Bayes-Stein with Shortsale Constraints (BS-C)
    strategy_weights["BS-C"] = bayes_stein_optimization_constrained(mean_returns, cov_matrix, rf_avg)

    # Strategy 11: Minimum-Variance with Shortsale Constraints (Min-C)
    strategy_weights["Min-C"] = minimum_variance_constrained(cov_matrix)

    # Strategy 12: Minimum-Variance with Generalized Constraints (G-Min-C)
    strategy_weights["G-Min-C"] = minimum_variance_generalized(cov_matrix, floor=0.05)

    # Strategy 13: Kan and Zhou’s Three-Fund Model (KZ)
    strategy_weights["KZ"] = kan_zhou_three_fund(mean_returns, cov_matrix, rf_avg, gamma=3.0, hedging_coef=-0.1)

    # Strategy 14: Mixture of Minimum-Variance and 1/N (EW-Min)
    strategy_weights["EW-Min"] = mixture_minvar_equal(cov_matrix, lambda_mix=0.5)

    # Strategy 15: Garlappi, Uppal, and Wang’s Model (GUW)
    strategy_weights["GUW"] = garlappi_uppal_wang(mean_returns, cov_matrix, rf_avg, tau=0.1)

    # =============================================================================
    # Section 6: Compute Portfolio Returns and Performance for Each Strategy
    # =============================================================================
    # Dictionary to hold performance metrics for each strategy
    performance_results = {}

    # For each strategy, compute the portfolio returns (using raw asset returns)
    # and then compute performance metrics.
    for strat_name, weights in strategy_weights.items():
        # Calculate daily portfolio returns as the weighted sum of asset returns.
        port_returns = (sector_returns * weights).sum(axis=1)
        cum_ret, ann_ret, ann_vol, sharpe = compute_performance(port_returns, rf_series)
        # Compute Certainty-Equivalent Return (CEQ) using the risk-aversion parameter gamma_risk.
        ceq = compute_CEQ(ann_ret, ann_vol, gamma_risk)
        # Store results in the performance dictionary.
        performance_results[strat_name] = {
            "Cumulative Return": cum_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "CEQ": ceq,
            "Turnover": np.nan  # Turnover not computed (requires dynamic rebalancing)
        }

    # =============================================================================
    # Section 7: Aggregate Results into a Table and Export to CSV
    # =============================================================================
    results_df = pd.DataFrame(performance_results).T
    results_df.index.name = "Strategy"
    results_df = results_df.reset_index()

    print("\nPortfolio Strategy Performance Metrics:")
    print(results_df)

    # Export the results to a CSV file for easy import into Excel
    results_df.to_csv("Portfolio Optimization/1. Ten sector portfolios of the S&P 500 and the US equity market portfolio/portfolio_results.csv", index=False)
    print("\nResults exported to 'portfolio_results.csv'")
