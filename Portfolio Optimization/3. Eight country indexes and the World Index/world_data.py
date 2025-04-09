#!/usr/bin/env python3
"""
Portfolio Strategies for World Data (Monthly)
------------------------------------------------
This script reads world index data from multiple CSV files (each file contains monthly
index levels), computes monthly returns, and then implements 15 portfolio allocation
strategies. The strategies include the naïve equal–weight (1/N) benchmark, simple
sample–based mean–variance, several Bayesian approaches, minimum–variance variants,
the Kan and Zhou three–fund model, a mixture of minimum–variance and equal–weight, and
the Garlappi, Uppal, and Wang model.

Performance metrics computed for each strategy (on a monthly basis) include:
  - Cumulative Return
  - Annualized Return (using a 12–period year)
  - Annualized Volatility
  - Sharpe Ratio (using the monthly risk–free rate, resampled from Fama–French data)
  - Certainty–Equivalent Return (CEQ), computed as:
       CEQ = Annualized Return – (γ/2) * (Annualized Volatility)^2
  - Turnover is set to NaN (since these are full–sample static weights)

The results are printed and exported to "portfolio_world_results.csv".
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
# Adjust the parameters (e.g. skiprows) if your world index files differ in format.
# Each CSV file is expected to have at least two columns:
#    Date, Value
# where Date is in a standard format (e.g. "1998-12-31") and Value is the index level.
# =============================================================================
def get_rf_data(file_path):
    """
    Extract risk-free rate data from the Fama–French CSV.
    The CSV is assumed to have data starting at row 5 with columns:
      Date, Mkt-RF, SMB, HML, RF
    """
    ff_data = pd.read_csv(file_path, skiprows=5, header=None,
                          names=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    # Keep only rows where the Date appears to be an 8-digit string
    ff_data = ff_data[ff_data["Date"].astype(str).str.match(r"^\d{8}$")]
    ff_data["Date"] = pd.to_datetime(ff_data["Date"], format="%Y%m%d", errors="coerce")
    ff_data = ff_data.dropna(subset=["Date"])
    return ff_data[["Date", "RF"]]

def load_world_data(file_path):
    """
    Load a world index CSV file (with monthly index levels) and compute monthly returns.
    This version assumes the CSV file has a header row with columns "Date" and "Value".
    If your files include extra header rows, adjust or add a skiprows parameter.
    """
    # Read CSV; adjust skiprows if your file contains extra header information.
    df = pd.read_csv(file_path)  # Assuming the first row is the header (i.e. "Date,Value")
    # Ensure there is a 'Date' column
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    # Convert the Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    # Replace any missing-value markers with NaN and drop missing rows
    df.replace([-99.99, -999], np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    # Assume the index level is in the first column (typically named "Value")
    level_series = df.iloc[:, 0]
    # Compute monthly returns from index levels
    monthly_returns = level_series.pct_change().dropna()
    # Convert index to monthly period and then back to month-end timestamps
    monthly_returns.index = monthly_returns.index.to_period('M').to_timestamp(how='end')
    monthly_returns = monthly_returns[~monthly_returns.index.duplicated(keep='first')]
    ret_df = monthly_returns.to_frame()
    # Name the column using the file name (last part of the file path)
    ret_df.columns = [file_path.split('/')[-1]]
    return ret_df

# =============================================================================
# Section 3: Performance Metrics Functions (Monthly)
# =============================================================================
def compute_monthly_performance(portfolio_returns, rf_series, periods_per_year=12):
    """
    Compute performance metrics for a portfolio based on monthly returns.
      - Cumulative Return
      - Annualized Return (using periods_per_year, e.g., 12 for monthly data)
      - Annualized Volatility
      - Sharpe Ratio (using the average monthly risk-free rate)
    """
    port_returns = portfolio_returns.dropna()
    cumulative_return = (1 + port_returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(port_returns)) - 1
    annualized_vol = port_returns.std() * np.sqrt(periods_per_year)
    sharpe = (annualized_return - rf_series.mean()) / annualized_vol
    return cumulative_return, annualized_return, annualized_vol, sharpe

def compute_CEQ(annualized_return, annualized_vol, gamma):
    """
    Compute the Certainty–Equivalent Return (CEQ) for a mean–variance investor.
    Formula:
        CEQ = Annualized Return - (gamma/2) * (Annualized Volatility)^2
    """
    return annualized_return - (gamma / 2) * (annualized_vol ** 2)

# =============================================================================
# Section 4: Portfolio Optimization Functions (15 Strategies)
#
# The following functions implement each of the 15 strategies.
# They are nearly identical to those used for daily data but are applied here
# to the monthly returns of world indices.
# =============================================================================

# Strategy 1: Naïve Diversification (1/N)
def equal_weight_portfolio(n_assets):
    """Return an equal–weight vector (each asset gets 1/n_assets)."""
    return np.ones(n_assets) / n_assets

# Strategy 2: Simple Sample–Based Mean–Variance (MV) (Unconstrained)
def mean_variance_closed_form(mean_returns, cov_matrix):
    """
    Compute the unconstrained mean–variance portfolio (closed–form solution).
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

# Strategy 3: Bayesian Diffuse–Prior Portfolio
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

# Strategy 5: Bayesian Data–and–Model (DM) Portfolio
def bayesian_data_and_model_portfolio(mean_returns, T):
    """
    Combine the sample mean with the global mean using Bayesian weighting.
    DM_mean = (T/(T+1)) * sample_mean + (1/(T+1)) * global_mean.
    """
    global_mean = mean_returns.mean()
    lambda_ = T / (T + 1)
    dm_mean = lambda_ * mean_returns + (1 - lambda_) * global_mean
    return dm_mean

# Strategy 6: Minimum–Variance (Min) Portfolio
def minimum_variance_portfolio(cov_matrix):
    """
    Compute the minimum–variance portfolio (ignoring expected returns).
    Formula:
         w_min = inv(cov_matrix) * ones / (1^T * inv(cov_matrix) * ones)
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    weights = inv_cov.dot(ones) / (ones.T.dot(inv_cov.dot(ones)))
    return weights

# Strategy 7: Value–Weighted Market Portfolio (VW)
def value_weighted_market_portfolio(tickers):
    """
    If 'SPY' is among the tickers (assumed to represent the market portfolio),
    assign 100% weight to SPY; otherwise, default to equal weights.
    """
    if "SPY" in tickers:
        weights = np.zeros(len(tickers))
        weights[tickers.index("SPY")] = 1.0
    else:
        weights = equal_weight_portfolio(len(tickers))
    return weights

# Strategy 8: MacKinlay and Pastor’s Missing–Factor Model (MP)
def mackinlay_pastor_portfolio(mean_returns, cov_matrix):
    """
    A placeholder for MacKinlay and Pastor's missing–factor model.
    For demonstration, return the simple MV closed–form weights.
    """
    return mean_variance_closed_form(mean_returns, cov_matrix)

# Strategy 9: MV with Shortsale Constraints (MV-C)
def mean_variance_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    Solve the mean–variance optimization with no short-selling constraints.
    Uses numerical optimization (SLSQP) with weights in [0,1] summing to 1.
    Objective: maximize Sharpe Ratio = (w^T * mean_returns - r_f) / portfolio volatility.
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
    First apply Bayes–Stein shrinkage to the sample mean, then solve for the
    mean–variance portfolio with no short-selling constraints.
    """
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    weights = mean_variance_optimization_constrained(bs_mean, cov_matrix, risk_free_rate)
    return weights

# Strategy 11: Minimum–Variance with Shortsale Constraints (Min-C)
def minimum_variance_constrained(cov_matrix):
    """
    Compute the minimum–variance portfolio with no short-selling.
    Uses numerical optimization with weights constrained between 0 and 1.
    """
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 12: Minimum–Variance with Generalized Constraints (G-Min-C)
def minimum_variance_generalized(cov_matrix, floor=0.05):
    """
    Compute the minimum–variance portfolio with additional floor constraints,
    requiring each weight to be at least 'floor'.
    """
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    # For each asset, ensure weight >= floor
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] - floor})
    bounds = [(None, None)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Strategy 13: Kan and Zhou’s Three–Fund Model (KZ)
def kan_zhou_three_fund(mean_returns, cov_matrix, risk_free_rate, gamma=3.0, hedging_coef=-0.1):
    """
    Implements a stylized three–fund separation model:
      1. Compute the myopic component: inv(cov_matrix) * (mean_returns - risk_free_rate)
      2. Compute the hedging component: inv(cov_matrix) * (mean_returns - BS_shrunk_mean)
      3. Combine: weights = (1/gamma) * myopic + hedging_coef * hedging
      4. Normalize the weights by dividing by the sum of absolute weights.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    myopic = inv_cov.dot(mean_returns - risk_free_rate)
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    hedging = inv_cov.dot(mean_returns - bs_mean)
    weights = (1 / gamma) * myopic + hedging_coef * hedging
    weights = weights / np.sum(np.abs(weights))
    return weights

# Strategy 14: Mixture of Minimum–Variance and 1/N (EW-Min)
def mixture_minvar_equal(cov_matrix, lambda_mix=0.5):
    """
    Compute a convex combination of the minimum–variance portfolio and the equal–weight portfolio.
    lambda_mix is the weight given to the minimum–variance portfolio.
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
         adjusted_mean = (1-tau)*sample_mean + tau*risk_free_rate,
    then compute the mean–variance closed–form portfolio.
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
    # List of world index file paths (adjust these paths as needed)
    world_file_paths = [
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/702743 - MSCI Italy 25_50 Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/912400 - MSCI Canada Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/925000 - MSCI France Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/928000 - MSCI Germany Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/975600 - MSCI Switzerland Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/939200 - MSCI Japan Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/982600 - MSCI UK Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/984000 - MSCI USA Index.csv",
        "Portfolio Optimization/3. Eight country indexes and the World Index/World Index/990100 - MSCI World Index.csv"
    ]
    # Fama-French file for risk–free rate
    ff_file_path = "Portfolio Optimization/1. Ten sector portfolios of the S&P 500 and the US equity market portfolio/F-F_Research_Data_Factors_daily.CSV"

    # ------------------------------
    # Load and Combine World Data
    # ------------------------------
    world_data_list = []
    for file_path in world_file_paths:
        try:
            wdata = load_world_data(file_path)
            world_data_list.append(wdata)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    try:
        combined_world = pd.concat(world_data_list, axis=1, join='inner')
    except Exception as e:
        print("Error concatenating world data:", e)
        exit()

    # ------------------------------
    # Prepare Monthly Risk-Free Rate
    # ------------------------------
    # Load daily risk–free rate data and resample to month–end
    rf_data_sp = get_rf_data(ff_file_path)
    rf_world = rf_data_sp.set_index("Date")["RF"].resample("ME").last()
    # Note: "ME" indicates month–end frequency

    # ------------------------------
    # Compute Sample Statistics
    # ------------------------------
    mean_returns_world = combined_world.mean()
    cov_matrix_world = combined_world.cov()

    # ------------------------------
    # Set Parameters for Performance Metrics
    # ------------------------------
    n_assets_world = combined_world.shape[1]
    T = len(combined_world)         # Sample size (number of monthly observations)
    rf_avg = rf_world.mean()          # Average monthly risk–free rate
    gamma_risk = 3.0                # Risk–aversion parameter for CEQ

    # ------------------------------
    # Compute Portfolio Weights for Each Strategy
    # ------------------------------
    strategy_weights = {}

    # Strategy 1: Naïve Diversification (1/N)
    strategy_weights["1/N"] = equal_weight_portfolio(n_assets_world)

    # Strategy 2: Simple Sample-Based Mean–Variance (MV)
    strategy_weights["MV"] = mean_variance_closed_form(mean_returns_world, cov_matrix_world)

    # Strategy 3: Bayesian Diffuse–Prior Portfolio
    diffused_mean = bayesian_diffuse_prior(mean_returns_world, T)
    strategy_weights["Bayesian Diffuse-Prior"] = mean_variance_closed_form(diffused_mean, cov_matrix_world)

    # Strategy 4: Bayes–Stein Portfolio
    bs_mean = bayes_stein_portfolio(mean_returns_world, cov_matrix_world)
    strategy_weights["Bayes-Stein"] = mean_variance_closed_form(bs_mean, cov_matrix_world)

    # Strategy 5: Bayesian Data–and–Model (DM) Portfolio
    dm_mean = bayesian_data_and_model_portfolio(mean_returns_world, T)
    strategy_weights["Bayesian Data-and-Model"] = mean_variance_closed_form(dm_mean, cov_matrix_world)

    # Strategy 6: Minimum–Variance (Min) Portfolio
    strategy_weights["Min"] = minimum_variance_portfolio(cov_matrix_world)

    # Strategy 7: Value–Weighted Market Portfolio (VW)
    # Use the column names from combined_world (if one of them is 'SPY')
    strategy_weights["VW"] = value_weighted_market_portfolio(list(combined_world.columns))

    # Strategy 8: MacKinlay and Pastor’s Missing–Factor Model (MP)
    strategy_weights["MP"] = mackinlay_pastor_portfolio(mean_returns_world, cov_matrix_world)

    # Strategy 9: MV with Shortsale Constraints (MV-C)
    strategy_weights["MV-C"] = mean_variance_optimization_constrained(mean_returns_world, cov_matrix_world, rf_avg)

    # Strategy 10: Bayes–Stein with Shortsale Constraints (BS-C)
    strategy_weights["BS-C"] = bayes_stein_optimization_constrained(mean_returns_world, cov_matrix_world, rf_avg)

    # Strategy 11: Minimum–Variance with Shortsale Constraints (Min-C)
    strategy_weights["Min-C"] = minimum_variance_constrained(cov_matrix_world)

    # Strategy 12: Minimum–Variance with Generalized Constraints (G-Min-C)
    strategy_weights["G-Min-C"] = minimum_variance_generalized(cov_matrix_world, floor=0.05)

    # Strategy 13: Kan and Zhou’s Three–Fund Model (KZ)
    strategy_weights["KZ"] = kan_zhou_three_fund(mean_returns_world, cov_matrix_world, rf_avg, gamma=3.0, hedging_coef=-0.1)

    # Strategy 14: Mixture of Minimum–Variance and 1/N (EW-Min)
    strategy_weights["EW-Min"] = mixture_minvar_equal(cov_matrix_world, lambda_mix=0.5)

    # Strategy 15: Garlappi, Uppal, and Wang’s (GUW) Model
    strategy_weights["GUW"] = garlappi_uppal_wang(mean_returns_world, cov_matrix_world, rf_avg, tau=0.1)

    # ------------------------------
    # Compute Portfolio Returns and Performance Metrics for Each Strategy
    # ------------------------------
    performance_results = {}

    for strat_name, weights in strategy_weights.items():
        # Calculate monthly portfolio returns as the weighted sum of the world index returns.
        port_returns = (combined_world * weights).sum(axis=1)
        cum_ret, ann_ret, ann_vol, sharpe = compute_monthly_performance(port_returns, rf_world)
        # Compute Certainty–Equivalent Return (CEQ)
        ceq = compute_CEQ(ann_ret, ann_vol, gamma_risk)
        # Turnover is set to NaN because we have static weights.
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

    print("\nCombined World Data Strategy Performance (Monthly):")
    print(results_df)

    results_df.to_csv("Portfolio Optimization/3. Eight country indexes and the World Index/portfolio_world_results.csv", index=False)
    print("\nResults exported to 'portfolio_world_results.csv'")
