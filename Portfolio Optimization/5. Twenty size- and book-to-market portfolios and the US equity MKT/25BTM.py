#!/usr/bin/env python3
"""
Portfolio Strategies for Dataset 5: Twenty Size- and Book-to-Market Portfolios 
and the US Equity Market
---------------------------------------------------------------------------
This script reads a CSV file containing daily average value–weighted returns.
The CSV file structure is as follows:
  - Row 18 contains a title/label (to be skipped).
  - Row 19 contains the column headers.
  - Data starts on row 20.
  
The first column is the date (in YYYYMMDD format) and the following 25 columns are 
the returns for 25 portfolios. The return values are expressed in percentage–points 
(e.g. -0.46 means -0.46%) and are converted to decimals (i.e. -0.0046) by dividing by 100.

The script computes sample statistics, applies 15 portfolio allocation strategies,
computes daily performance metrics (cumulative return, annualized return, annualized volatility,
Sharpe ratio, and Certainty–Equivalent Return [CEQ]), and exports the results to a CSV file.
Turnover is not computed (set to NaN) because we compute static, full–sample weights.

Adjust the file path and parameters (such as skip_rows) as needed.
"""

# =============================================================================
# Section 1: Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# =============================================================================
# Section 2: Data Loading and Preprocessing for Dataset 5
# =============================================================================
def load_dataset5(file_path, skip_rows=18, convert_to_decimal=True):
    """
    Load Dataset 5 from a CSV file.
    
    Parameters:
      - file_path (str): Path to the CSV file.
      - skip_rows (int): Number of rows to skip so that the header row (row 19)
                         is read as the header. (Here, skip_rows=18.)
      - convert_to_decimal (bool): If True, divide return values by 100.
    
    Returns:
      - DataFrame with a DateTime index and 25 columns of daily returns.
      
    We supply our own header names because the CSV’s header row does not include a 
    column name for the date.
    """
    # Define header names: first column as "Date", then 25 portfolio names.
    col_names = ["Date", "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
                 "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
                 "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
                 "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
                 "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM"]
    
    # Read the CSV file; skip the first skip_rows rows so that row 19 is treated as header.
    df = pd.read_csv(file_path, skiprows=skip_rows, header=0, names=col_names, low_memory=False)
    
    # Debug: Uncomment to inspect the first few rows
    # print(df.head(5))
    
    # Convert the "Date" column to datetime.
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)
    
    # Force all portfolio columns to be numeric.
    for col in col_names[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Optionally, convert returns from percentage–points to decimals.
    if convert_to_decimal:
        df = df / 100.0
    
    # Drop any rows that are completely NaN.
    df = df.dropna(how="all")
    return df

# =============================================================================
# Section 3: Performance Metrics Functions (Daily)
# =============================================================================
def compute_performance(portfolio_returns, rf_rate=0.0):
    """
    Compute performance metrics from daily portfolio returns.
    
    Metrics:
      - Cumulative Return: computed as exp(sum(log(1 + returns))) - 1
      - Annualized Return: compounded over 252 trading days.
      - Annualized Volatility: standard deviation scaled by sqrt(252)
      - Sharpe Ratio: (Annualized Return - rf_rate) / Annualized Volatility
      
    Using log returns avoids numerical overflow in computing the product.
    
    Parameters:
      - portfolio_returns (Series): daily portfolio returns.
      - rf_rate (float): daily risk–free rate (default 0.0).
      
    Returns:
      Tuple: (cumulative_return, annualized_return, annualized_volatility, sharpe_ratio)
    """
    returns = portfolio_returns.dropna()
    # Use log returns for numerical stability.
    cumulative_return = np.expm1(np.log1p(returns).sum())
    annualized_return = np.expm1(np.log1p(returns).mean() * 252)
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = (annualized_return - rf_rate) / annualized_vol if annualized_vol != 0 else np.nan
    return cumulative_return, annualized_return, annualized_vol, sharpe

def compute_CEQ(annualized_return, annualized_vol, gamma):
    """
    Compute the Certainty–Equivalent Return (CEQ) for a mean–variance investor.
    
    Formula:
      CEQ = Annualized Return - (gamma/2) * (Annualized Volatility)^2
      
    Parameters:
      - annualized_return (float)
      - annualized_vol (float)
      - gamma (float): risk–aversion parameter.
      
    Returns:
      float: CEQ value.
    """
    return annualized_return - (gamma / 2) * (annualized_vol ** 2)

# =============================================================================
# Section 4: Portfolio Optimization Functions (15 Strategies)
# =============================================================================
# The following functions implement the 15 strategies. They are similar to our other files.

def equal_weight_portfolio(n_assets):
    """Return an equal–weight portfolio (each asset gets 1/n_assets)."""
    return np.ones(n_assets) / n_assets

def mean_variance_closed_form(mean_returns, cov_matrix):
    """
    Compute the unconstrained mean–variance portfolio (closed–form).
    
    Formula:
      w = inv(cov_matrix) * mean_returns / (1^T * inv(cov_matrix) * mean_returns)
      
    Allows short-selling.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_returns))
    weights = inv_cov.dot(mean_returns) / ones.T.dot(inv_cov.dot(mean_returns))
    return weights

def bayesian_diffuse_prior(mean_returns, T):
    """Return the diffuse–prior adjusted mean: (T/(T+1))*sample_mean."""
    return (T / (T + 1)) * mean_returns

def bayes_stein_portfolio(mean_returns, cov_matrix):
    """Return the Bayes–Stein shrunk mean estimate."""
    n = len(mean_returns)
    global_mean = mean_returns.mean()
    total_variance = np.trace(cov_matrix) / n
    sample_variance = mean_returns.var()
    phi = sample_variance / (sample_variance + total_variance)
    return global_mean + phi * (mean_returns - global_mean)

def bayesian_data_and_model_portfolio(mean_returns, T):
    """Return the DM adjusted mean: (T/(T+1))*sample_mean + (1/(T+1))*global_mean."""
    global_mean = mean_returns.mean()
    lambda_ = T / (T + 1)
    return lambda_ * mean_returns + (1 - lambda_) * global_mean

def minimum_variance_portfolio(cov_matrix):
    """Return the minimum–variance portfolio (ignoring expected returns)."""
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    return inv_cov.dot(ones) / (ones.T.dot(inv_cov.dot(ones)))

def value_weighted_market_portfolio(tickers):
    """
    If 'SPY' is among the tickers (representing the US market), assign 100% weight to SPY;
    otherwise, use equal weighting.
    """
    if "SPY" in tickers:
        weights = np.zeros(len(tickers))
        weights[tickers.index("SPY")] = 1.0
    else:
        weights = equal_weight_portfolio(len(tickers))
    return weights

def mackinlay_pastor_portfolio(mean_returns, cov_matrix):
    """Return the MV closed–form portfolio (placeholder for MP model)."""
    return mean_variance_closed_form(mean_returns, cov_matrix)

def mean_variance_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0001):
    """
    Compute the MV portfolio with no short-selling (weights in [0,1]) via numerical optimization.
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
    """Return the constrained MV portfolio after applying Bayes–Stein shrinkage."""
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    return mean_variance_optimization_constrained(bs_mean, cov_matrix, risk_free_rate)

def minimum_variance_constrained(cov_matrix):
    """
    Compute the minimum–variance portfolio with no short-selling constraints.
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
    Compute the minimum–variance portfolio with a floor constraint on each weight.
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
    Return portfolio weights using Kan and Zhou’s three–fund model.
      1. Myopic component: inv(cov_matrix) * (mean_returns - risk_free_rate)
      2. Hedging component: inv(cov_matrix) * (mean_returns - BS_shrunk_mean)
      3. Combined weights: (1/gamma)*myopic + hedging_coef*hedging, normalized by sum of absolute weights.
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
    Return a convex combination of the minimum–variance portfolio and the equal–weight portfolio.
    """
    w_min = minimum_variance_portfolio(cov_matrix)
    w_equal = equal_weight_portfolio(len(cov_matrix))
    return lambda_mix * w_min + (1 - lambda_mix) * w_equal

def garlappi_uppal_wang(mean_returns, cov_matrix, risk_free_rate, tau=0.1):
    """
    Return portfolio weights using a simplified version of the GUW model.
      adjusted_mean = (1 - tau)*sample_mean + tau*risk_free_rate,
      then compute the MV closed–form portfolio.
    """
    adjusted_mean = (1 - tau) * mean_returns + tau * risk_free_rate
    return mean_variance_closed_form(adjusted_mean, cov_matrix)

# =============================================================================
# Section 5: Main Execution – Process Dataset 5, Compute Strategies, and Export Results
# =============================================================================
if __name__ == "__main__":
    # ------------------------------
    # Data Input for Dataset 5
    # ------------------------------
    dataset5_file = "Portfolio Optimization/5. Twenty size- and book-to-market portfolios and the US equity MKT/25_Portfolios_5x5_Daily.csv"
    
    # Load Dataset 5. We skip the first 18 rows so that row 19 is the header,
    # and data starts on row 20.
    df = load_dataset5(dataset5_file, skip_rows=18, convert_to_decimal=True)
    # Debug: Uncomment the next line to inspect the first few rows.
    # print(df.head())
    
    # ------------------------------
    # Compute Sample Statistics for Dataset 5
    # ------------------------------
    mean_returns = df.mean()
    cov_matrix = df.cov()
    
    # Assume no separate risk-free rate is provided (set to 0)
    rf_rate = 0.0
    n_assets = len(df.columns)  # should be 25
    T = len(df)                 # number of trading days
    gamma_risk = 3.0            # risk-aversion parameter for CEQ
    
    # ------------------------------
    # Compute Portfolio Weights for Each of the 15 Strategies
    # ------------------------------
    strategy_weights = {}
    strategy_weights["1/N"] = equal_weight_portfolio(n_assets)
    strategy_weights["MV"] = mean_variance_closed_form(mean_returns, cov_matrix)
    diffused_mean = bayesian_diffuse_prior(mean_returns, T)
    strategy_weights["Bayesian Diffuse-Prior"] = mean_variance_closed_form(diffused_mean, cov_matrix)
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    strategy_weights["Bayes-Stein"] = mean_variance_closed_form(bs_mean, cov_matrix)
    dm_mean = bayesian_data_and_model_portfolio(mean_returns, T)
    strategy_weights["Bayesian Data-and-Model"] = mean_variance_closed_form(dm_mean, cov_matrix)
    strategy_weights["Min"] = minimum_variance_portfolio(cov_matrix)
    strategy_weights["VW"] = value_weighted_market_portfolio(list(df.columns))
    strategy_weights["MP"] = mackinlay_pastor_portfolio(mean_returns, cov_matrix)
    strategy_weights["MV-C"] = mean_variance_optimization_constrained(mean_returns, cov_matrix, rf_rate)
    strategy_weights["BS-C"] = bayes_stein_optimization_constrained(mean_returns, cov_matrix, rf_rate)
    strategy_weights["Min-C"] = minimum_variance_constrained(cov_matrix)
    strategy_weights["G-Min-C"] = minimum_variance_generalized(cov_matrix, floor=0.05)
    strategy_weights["KZ"] = kan_zhou_three_fund(mean_returns, cov_matrix, rf_rate, gamma=3.0, hedging_coef=-0.1)
    strategy_weights["EW-Min"] = mixture_minvar_equal(cov_matrix, lambda_mix=0.5)
    strategy_weights["GUW"] = garlappi_uppal_wang(mean_returns, cov_matrix, rf_rate, tau=0.1)
    
    # ------------------------------
    # Compute Portfolio Returns and Performance Metrics for Each Strategy
    # ------------------------------
    performance_results = {}
    for strat_name, weights in strategy_weights.items():
        # Compute daily portfolio returns as the weighted sum across all 25 portfolios.
        port_returns = (df * weights).sum(axis=1)
        # Use our log-return based computation for cumulative return to avoid overflow.
        cum_ret, ann_ret, ann_vol, sharpe = compute_performance(port_returns, rf_rate)
        ceq = compute_CEQ(ann_ret, ann_vol, gamma_risk)
        performance_results[strat_name] = {
            "Cumulative Return": cum_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "CEQ": ceq,
            "Turnover": np.nan
        }
    
    # ------------------------------
    # Aggregate the Results into a DataFrame and Export to CSV
    # ------------------------------
    results_df = pd.DataFrame(performance_results).T
    results_df.index.name = "Strategy"
    results_df = results_df.reset_index()
    
    print("\nDataset 5 Portfolio Strategy Performance (Daily):")
    print(results_df)
    
    results_df.to_csv("portfolio_dataset5_results.csv", index=False)
    print("\nResults exported to 'portfolio_dataset5_results.csv'")
