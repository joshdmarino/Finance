#!/usr/bin/env python3
"""
Portfolio Strategies Comparison Script for Combined Industry + SPY Data
------------------------------------------------------------------------
This script implements 15 portfolio allocation strategies (including the naïve 
1/N benchmark, simple mean-variance, and 13 additional strategies) on a dataset 
that combines the 10 Industry Portfolios with SPY returns.

The script computes the following performance metrics for each strategy:
  - Cumulative Return
  - Annualized Return (assumes 252 trading days per year)
  - Annualized Volatility
  - Sharpe Ratio
  - Certainty-Equivalent Return (CEQ)
  - Turnover (set as NaN because this is a static, full-sample allocation)

The CEQ is computed using the formula:
    CEQ = Annualized Return - (γ/2) * (Annualized Volatility)^2
where γ is a risk-aversion parameter (set here to 3.0 by default).

The resulting performance table is printed and exported to a CSV file for Excel.
"""

# =============================================================================
# Section 1: Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# =============================================================================
# Section 2: Data Loading and Preprocessing Functions
#
# These functions read the necessary datasets.
# You can modify the file paths and parameters in the main section.
# =============================================================================
def get_rf_data(file_path):
    """
    Extract risk-free rate data from the Fama-French CSV.
    The CSV is assumed to have data starting at row 5 with columns:
      Date, Mkt-RF, SMB, HML, RF
    """
    ff_data = pd.read_csv(file_path, skiprows=5, header=None,
                          names=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    # Keep only rows with proper 8-digit dates
    ff_data = ff_data[ff_data["Date"].astype(str).str.match(r"^\d{8}$")]
    ff_data["Date"] = pd.to_datetime(ff_data["Date"], format="%Y%m%d", errors="coerce")
    ff_data = ff_data.dropna(subset=["Date"])
    return ff_data[["Date", "RF"]]

def load_industry_data(file_path):
    """
    Load the 10 Industry Portfolios data from a CSV file.
    The file is assumed to contain a 'Date' column and 10 portfolio columns.
    The data is converted from percentage points to decimals.
    """
    industry_data = pd.read_csv(file_path, skiprows=11, header=0)
    if industry_data.columns[0] != 'Date':
        industry_data.rename(columns={industry_data.columns[0]: 'Date'}, inplace=True)
    industry_data['Date'] = pd.to_datetime(industry_data['Date'], format='%Y%m%d', errors='coerce')
    industry_data.replace([-99.99, -999], np.nan, inplace=True)
    industry_data.set_index('Date', inplace=True)
    industry_data = industry_data.apply(pd.to_numeric, errors='coerce').dropna()
    # Convert from percentage points to decimals
    industry_data = industry_data / 100
    try:
        float(industry_data.columns[0])
        expected_names = ['Consumer Durables', 'Consumer NonDurables', 'Manufacturing', 'Energy', 
                          'Business Equipment', 'Telecom', 'Shops', 'Health', 'Utilities', 'Other']
        industry_data.columns = expected_names
    except ValueError:
        pass
    return industry_data

def load_spy_data(start_date, end_date):
    """
    Download SPY adjusted close prices from Yahoo Finance and compute daily returns.
    """
    spy_data = yf.download("SPY", start=start_date, end=end_date)["Adj Close"]
    spy_returns = spy_data.pct_change().dropna()
    return spy_returns

# =============================================================================
# Section 3: Performance Metrics Functions
# =============================================================================
def compute_performance(portfolio_returns, rf_series):
    """
    Compute performance metrics based on daily portfolio returns:
      - Cumulative Return
      - Annualized Return (using 252 trading days)
      - Annualized Volatility
      - Sharpe Ratio (using average risk-free rate)
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
    """
    return annualized_return - (gamma / 2) * (annualized_vol ** 2)

# =============================================================================
# Section 4: Portfolio Optimization Functions
#
# The following functions implement the 15 strategies.
# Detailed comments describe each method's logic and constraints.
# =============================================================================

# Strategy 1: Naïve Diversification (1/N)
def equal_weight_portfolio(n_assets):
    """Return an equal-weight vector (each asset gets 1/n_assets)."""
    return np.ones(n_assets) / n_assets

# Strategy 2: Simple Sample-Based Mean-Variance (MV) (Unconstrained)
def mean_variance_closed_form(mean_returns, cov_matrix):
    """
    Compute the unconstrained mean-variance portfolio using the closed-form solution.
    Formula:
        w_MV = inv(cov_matrix) * mean_returns / (1^T * inv(cov_matrix) * mean_returns)
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
    Adjust the sample mean by assuming a diffuse (flat) prior.
    diffused_mean = (T/(T+1)) * sample_mean, where T is the sample size.
    """
    return (T / (T + 1)) * mean_returns

# Strategy 4: Bayes–Stein Portfolio
def bayes_stein_portfolio(mean_returns, cov_matrix):
    """
    Apply Bayes–Stein shrinkage to the sample mean.
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
    Combine the sample mean with the global mean using Bayesian weighting.
    DM_mean = (T/(T+1)) * sample_mean + (1/(T+1)) * global_mean.
    """
    global_mean = mean_returns.mean()
    lambda_ = T / (T + 1)
    dm_mean = lambda_ * mean_returns + (1 - lambda_) * global_mean
    return dm_mean

# Strategy 6: Minimum-Variance (Min) Portfolio
def minimum_variance_portfolio(cov_matrix):
    """
    Compute the minimum-variance portfolio (does not use expected returns).
    Formula:
        w_min = inv(cov_matrix) * ones / (1^T * inv(cov_matrix) * ones)
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    weights = inv_cov.dot(ones) / (ones.T.dot(inv_cov.dot(ones)))
    return weights

# Strategy 7: Value-Weighted Market Portfolio (VW)
def value_weighted_market_portfolio(tickers):
    """
    If 'SPY' is in the tickers (assumed to represent the market portfolio),
    assign 100% weight to SPY; otherwise, default to equal weights.
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
    A placeholder for MacKinlay and Pastor's missing-factor model.
    For demonstration purposes, we return the simple MV closed-form weights.
    """
    return mean_variance_closed_form(mean_returns, cov_matrix)

# Strategy 9: MV with Shortsale Constraints (MV-C)
def mean_variance_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    Solve the mean-variance optimization with no short-selling (weights constrained between 0 and 1).
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

# Strategy 10: Bayes–Stein with Shortsale Constraints (BS-C)
def bayes_stein_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    First apply Bayes–Stein shrinkage to the sample mean, then solve for the MV portfolio
    with no short-selling constraints.
    """
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    weights = mean_variance_optimization_constrained(bs_mean, cov_matrix, risk_free_rate)
    return weights

# Strategy 11: Minimum-Variance with Shortsale Constraints (Min-C)
def minimum_variance_constrained(cov_matrix):
    """
    Compute the minimum-variance portfolio subject to no short-selling.
    Uses numerical optimization with weights bounded between 0 and 1.
    """
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 12: Minimum-Variance with Generalized Constraints (G-Min-C)
def minimum_variance_generalized(cov_matrix, floor=0.05):
    """
    Compute the minimum-variance portfolio with additional floor constraints,
    requiring each weight to be at least 'floor'.
    """
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    # Add constraint for each weight: w[i] >= floor
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] - floor})
    bounds = [(None, None)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 13: Kan and Zhou’s Three-Fund Model (KZ)
def kan_zhou_three_fund(mean_returns, cov_matrix, risk_free_rate, gamma=3.0, hedging_coef=-0.1):
    """
    Implements a stylized three-fund separation model:
      1. Compute the myopic component: inv(cov_matrix) * (mean_returns - risk_free_rate)
      2. Compute the hedging component: inv(cov_matrix) * (mean_returns - BS_shrunk_mean)
      3. Combine them: weights = (1/gamma) * myopic + hedging_coef * hedging
      4. Normalize the weights (here, by the sum of absolute values).
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
    A simplified implementation of the GUW model.
    Adjust the sample mean by shrinking it toward the risk-free rate:
         adjusted_mean = (1 - tau)*sample_mean + tau*risk_free_rate,
    then compute the MV closed-form portfolio.
    """
    adjusted_mean = (1 - tau) * mean_returns + tau * risk_free_rate
    weights = mean_variance_closed_form(adjusted_mean, cov_matrix)
    return weights

# =============================================================================
# Section 5: Main Execution - Data Preparation, Strategy Computation, and Results
# =============================================================================
if __name__ == "__main__":
    # ------------------------------
    # Data Input Parameters
    # ------------------------------
    ff_file_path = "Portfolio Optimization/1. Ten sector portfolios of the S&P 500 and the US equity market portfolio/F-F_Research_Data_Factors_daily.CSV"
    industry_file_path = "Portfolio Optimization/2. Ten industry portfolios and the US equity market portfolio/10_Industry_Portfolios_Daily.csv"
    spy_start_date = "1984-01-01"
    spy_end_date   = "2024-12-31"
    
    # ------------------------------
    # Load Combined Industry and SPY Data
    # ------------------------------
    # Load the 10 Industry Portfolios
    industry_data = load_industry_data(industry_file_path)
    # Load SPY returns
    spy_returns = load_spy_data(spy_start_date, spy_end_date)
    # Merge the two datasets on their dates (inner join on the index)
    combined_ind_spy = pd.merge(industry_data, spy_returns, left_index=True, right_index=True, how='inner')
    # Rename the SPY column if necessary
    if "Adj Close" in combined_ind_spy.columns:
        combined_ind_spy.rename(columns={"Adj Close": "SPY"}, inplace=True)
    elif "SPY" not in combined_ind_spy.columns:
        combined_ind_spy.rename(columns={combined_ind_spy.columns[-1]: "SPY"}, inplace=True)
    
    print("=== Combined Industry + SPY Data ===")
    print("Date Range:", combined_ind_spy.index.min().date(), "to", combined_ind_spy.index.max().date())
    print(combined_ind_spy.head(), "\n")
    
    # Load risk-free rate data and align with the combined dataset dates
    rf_data_sp = get_rf_data(ff_file_path)
    rf_combined = rf_data_sp.set_index("Date")["RF"].reindex(combined_ind_spy.index, method="ffill")
    
    # ------------------------------
    # Compute Sample Statistics
    # ------------------------------
    # Compute the sample mean and covariance matrix from the combined dataset.
    mean_returns_ind = combined_ind_spy.mean()
    cov_matrix_ind = combined_ind_spy.cov()
    
    # ------------------------------
    # Set Parameters for Performance Metrics
    # ------------------------------
    n_assets = len(mean_returns_ind)
    T = len(combined_ind_spy)         # Sample size (number of observations)
    rf_avg = rf_combined.mean()         # Average risk-free rate over the period
    gamma_risk = 3.0                  # Risk-aversion parameter for CEQ calculation
    
    # ------------------------------
    # Compute Portfolio Weights for Each Strategy
    # ------------------------------
    strategy_weights = {}
    
    # Strategy 1: Naïve Diversification (1/N)
    strategy_weights["1/N"] = equal_weight_portfolio(n_assets)
    
    # Strategy 2: Simple Sample-Based Mean-Variance (MV)
    strategy_weights["MV"] = mean_variance_closed_form(mean_returns_ind, cov_matrix_ind)
    
    # Strategy 3: Bayesian Diffuse-Prior Portfolio
    diffused_mean = bayesian_diffuse_prior(mean_returns_ind, T)
    strategy_weights["Bayesian Diffuse-Prior"] = mean_variance_closed_form(diffused_mean, cov_matrix_ind)
    
    # Strategy 4: Bayes–Stein Portfolio
    bs_mean = bayes_stein_portfolio(mean_returns_ind, cov_matrix_ind)
    strategy_weights["Bayes-Stein"] = mean_variance_closed_form(bs_mean, cov_matrix_ind)
    
    # Strategy 5: Bayesian Data-and-Model (DM) Portfolio
    dm_mean = bayesian_data_and_model_portfolio(mean_returns_ind, T)
    strategy_weights["Bayesian Data-and-Model"] = mean_variance_closed_form(dm_mean, cov_matrix_ind)
    
    # Strategy 6: Minimum-Variance (Min) Portfolio
    strategy_weights["Min"] = minimum_variance_portfolio(cov_matrix_ind)
    
    # Strategy 7: Value-Weighted Market Portfolio (VW)
    strategy_weights["VW"] = value_weighted_market_portfolio(list(combined_ind_spy.columns))
    
    # Strategy 8: MacKinlay and Pastor’s Missing-Factor Model (MP)
    strategy_weights["MP"] = mackinlay_pastor_portfolio(mean_returns_ind, cov_matrix_ind)
    
    # Strategy 9: MV with Shortsale Constraints (MV-C)
    strategy_weights["MV-C"] = mean_variance_optimization_constrained(mean_returns_ind, cov_matrix_ind, rf_avg)
    
    # Strategy 10: Bayes-Stein with Shortsale Constraints (BS-C)
    strategy_weights["BS-C"] = bayes_stein_optimization_constrained(mean_returns_ind, cov_matrix_ind, rf_avg)
    
    # Strategy 11: Minimum-Variance with Shortsale Constraints (Min-C)
    strategy_weights["Min-C"] = minimum_variance_constrained(cov_matrix_ind)
    
    # Strategy 12: Minimum-Variance with Generalized Constraints (G-Min-C)
    strategy_weights["G-Min-C"] = minimum_variance_generalized(cov_matrix_ind, floor=0.05)
    
    # Strategy 13: Kan and Zhou’s Three-Fund Model (KZ)
    strategy_weights["KZ"] = kan_zhou_three_fund(mean_returns_ind, cov_matrix_ind, rf_avg, gamma=3.0, hedging_coef=-0.1)
    
    # Strategy 14: Mixture of Minimum-Variance and 1/N (EW-Min)
    strategy_weights["EW-Min"] = mixture_minvar_equal(cov_matrix_ind, lambda_mix=0.5)
    
    # Strategy 15: Garlappi, Uppal, and Wang’s (GUW) Model
    strategy_weights["GUW"] = garlappi_uppal_wang(mean_returns_ind, cov_matrix_ind, rf_avg, tau=0.1)
    
    # ------------------------------
    # Compute Portfolio Returns and Performance Metrics for Each Strategy
    # ------------------------------
    performance_results = {}
    
    for strat_name, weights in strategy_weights.items():
        # Calculate daily portfolio returns as the weighted sum of asset returns.
        port_returns = (combined_ind_spy * weights).sum(axis=1)
        cum_ret, ann_ret, ann_vol, sharpe = compute_performance(port_returns, rf_combined)
        # Compute Certainty-Equivalent Return (CEQ) using the risk-aversion parameter.
        ceq = compute_CEQ(ann_ret, ann_vol, gamma_risk)
        # Turnover is set to NaN (since we have static full-sample weights).
        performance_results[strat_name] = {
            "Cumulative Return": cum_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "CEQ": ceq,
            "Turnover": np.nan
        }
    
    # ------------------------------
    # Aggregate the Results into a Table and Export to CSV
    # ------------------------------
    results_df = pd.DataFrame(performance_results).T
    results_df.index.name = "Strategy"
    results_df = results_df.reset_index()
    
    print("\nPortfolio Strategy Performance Metrics (Combined Industry + SPY):")
    print(results_df)
    
    results_df.to_csv("Portfolio Optimization/2. Ten industry portfolios and the US equity market portfolio/portfolio_ind_spy_results.csv", index=False)
    print("\nResults exported to 'portfolio_ind_spy_results.csv'")
