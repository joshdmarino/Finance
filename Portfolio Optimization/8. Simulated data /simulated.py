#!/usr/bin/env python3
"""
Simulated Data Portfolio Strategies Backtest
---------------------------------------------
This script simulates monthly excess returns for N assets using a factor–model:

    R_t = μ + β F_t + ε_t

Parameters (per month):
  - Time Period: 20 years = 240 months.
  - Number of Assets (N): 25 (change to 10 or 50 if desired).
  - Number of Factors (K): 3.
  - Expected Returns: μ_i ~ N(0.005, 0.02)
  - Factor Loadings: β_{i,k} ~ N(1, 0.2)
  - Factor Returns: F_t ~ N(0, (0.02)^2 I_K)
  - Idiosyncratic Noise: ε_t ~ N(0, (0.05)^2 I)
  
The asset return covariance is computed as:
  
    Σ = β Σ_F βᵀ + Σ_ε

After simulating 240 months of returns, the script uses a rolling-window
backtest with a 120–month estimation window (10 years) to compute portfolio weights
using 15 strategies. A small regularization is applied to the estimated covariance
to stabilize the optimization.

Performance metrics (for monthly data) include:
  - Cumulative Return
  - Annualized Return (using 12 months per year)
  - Annualized Volatility
  - Sharpe Ratio
  - Certainty–Equivalent Return (CEQ), with gamma_risk = 3.0
  - Average Turnover

Results are printed and exported to a CSV file.
"""

# =============================================================================
# Section 1: Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# Section 2: Simulation Parameters and Data Generation
# =============================================================================
# Simulation parameters
N = 25           # Number of assets (change to 10 or 50 if desired)
K = 3            # Number of factors
T_total = 240    # Total number of months (20 years)
L = 120          # Estimation window in months (10 years)
rf_rate = 0.0    # Risk-free rate (excess returns already simulated)

# Risk-aversion parameter for CEQ computation
gamma_risk = 3.0

# Set seed for reproducibility
np.random.seed(42)

# Generate true expected excess returns: μ_i ~ N(0.005, 0.02)
mu_true = np.random.normal(loc=0.005, scale=0.02, size=N)

# Generate factor loadings: β_{i,k} ~ N(1, 0.2)
beta_true = np.random.normal(loc=1.0, scale=0.2, size=(N, K))

# Factor return volatility: σ_F = 0.02 → variance = (0.02)^2
sigma_factor = 0.02
Sigma_F = np.diag([sigma_factor**2] * K)

# Idiosyncratic volatility: σ_ε = 0.05 → variance = (0.05)^2
sigma_eps = 0.05
Sigma_eps = np.diag([sigma_eps**2] * N)

# True covariance of asset returns: Σ = β Σ_F βᵀ + Σ_ε
Sigma_true = beta_true.dot(Sigma_F).dot(beta_true.T) + Sigma_eps

# Pre-allocate array for simulated returns (T_total x N)
simulated_returns = np.zeros((T_total, N))

# Simulate factor returns F_t ~ N(0, Σ_F) for each month (240 x K)
factor_returns = np.random.multivariate_normal(mean=np.zeros(K), cov=Sigma_F, size=T_total)

# Simulate idiosyncratic noise: ε_t ~ N(0, Σ_eps)
noise = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eps, size=T_total)

# Generate simulated asset returns for each month:
# R_t = μ + β F_t + ε_t
for t in range(T_total):
    simulated_returns[t, :] = mu_true + beta_true.dot(factor_returns[t, :]) + noise[t, :]

# Create a DataFrame of simulated returns with a monthly DateTime index.
dates = pd.date_range(start="2000-01-31", periods=T_total, freq="ME")
simulated_df = pd.DataFrame(simulated_returns, index=dates,
                            columns=[f"Asset_{i+1}" for i in range(N)])

# =============================================================================
# Section 2.1: Covariance Matrix Regularization
# =============================================================================
def regularize_cov(cov_matrix, epsilon=1e-4):
    """
    Regularize the covariance matrix by adding epsilon * I.
    This helps avoid numerical instabilities in matrix inversion.
    """
    return cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

# =============================================================================
# Section 3: Performance Metrics Functions
# =============================================================================
def compute_performance(portfolio_returns, rf_rate=0.0):
    """
    Compute performance metrics from portfolio returns (monthly data).

    Returns:
      - Cumulative Return: exp(sum(log(1 + returns))) - 1
      - Annualized Return: exp(mean(log(1 + returns)) * 12) - 1
      - Annualized Volatility: std * sqrt(12)
      - Sharpe Ratio: (Annualized Return - rf_rate) / Annualized Volatility
    """
    returns = portfolio_returns.dropna()
    cumulative_return = np.expm1(np.log1p(returns).sum())
    annualized_return = np.expm1(np.log1p(returns).mean() * 12) - 1
    annualized_vol = returns.std() * np.sqrt(12)
    sharpe = (annualized_return - rf_rate) / annualized_vol if annualized_vol != 0 else np.nan
    return cumulative_return, annualized_return, annualized_vol, sharpe

def compute_CEQ(annualized_return, annualized_vol, gamma):
    """
    Compute the Certainty–Equivalent Return (CEQ).

    CEQ = Annualized Return - (gamma/2) * (Annualized Volatility)^2
    """
    return annualized_return - (gamma / 2) * (annualized_vol ** 2)

# =============================================================================
# Section 4: Portfolio Optimization Functions (15 Strategies)
# =============================================================================
def equal_weight_portfolio(n_assets):
    return np.ones(n_assets) / n_assets

def mean_variance_closed_form(mean_returns, cov_matrix):
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_returns))
    return inv_cov.dot(mean_returns) / ones.T.dot(inv_cov.dot(mean_returns))

def bayesian_diffuse_prior(mean_returns, T):
    return (T / (T + 1)) * mean_returns

def bayes_stein_portfolio(mean_returns, cov_matrix):
    n = len(mean_returns)
    global_mean = mean_returns.mean()
    total_variance = np.trace(cov_matrix) / n
    sample_variance = mean_returns.var()
    phi = sample_variance / (sample_variance + total_variance)
    return global_mean + phi * (mean_returns - global_mean)

def bayesian_data_and_model_portfolio(mean_returns, T):
    global_mean = mean_returns.mean()
    lambda_ = T / (T + 1)
    return lambda_ * mean_returns + (1 - lambda_) * global_mean

def minimum_variance_portfolio(cov_matrix):
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    return inv_cov.dot(ones) / (ones.T.dot(inv_cov.dot(ones)))

def value_weighted_market_portfolio(tickers):
    return equal_weight_portfolio(len(tickers))

def mackinlay_pastor_portfolio(mean_returns, cov_matrix):
    return mean_variance_closed_form(mean_returns, cov_matrix)

def mean_variance_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0):
    n = len(mean_returns)
    def objective(w):
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -((port_return - risk_free_rate) / port_vol)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 100, 'disp': False})
    return result.x

def bayes_stein_optimization_constrained(mean_returns, cov_matrix, risk_free_rate=0.0):
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    return mean_variance_optimization_constrained(bs_mean, cov_matrix, risk_free_rate)

def minimum_variance_constrained(cov_matrix):
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 100, 'disp': False})
    return result.x

def minimum_variance_generalized(cov_matrix, floor=0.05):
    n = len(cov_matrix)
    def objective(w):
        return np.dot(w.T, np.dot(cov_matrix, w))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] - floor})
    bounds = [(None, None)] * n
    initial_guess = equal_weight_portfolio(n)
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 100, 'disp': False})
    return result.x

def kan_zhou_three_fund(mean_returns, cov_matrix, risk_free_rate, gamma=3.0, hedging_coef=-0.1):
    inv_cov = np.linalg.inv(cov_matrix)
    myopic = inv_cov.dot(mean_returns - risk_free_rate)
    bs_mean = bayes_stein_portfolio(mean_returns, cov_matrix)
    hedging = inv_cov.dot(mean_returns - bs_mean)
    weights = (1 / gamma) * myopic + hedging_coef * hedging
    return weights / np.sum(np.abs(weights))

def mixture_minvar_equal(cov_matrix, lambda_mix=0.5):
    w_min = minimum_variance_portfolio(cov_matrix)
    w_equal = equal_weight_portfolio(len(cov_matrix))
    return lambda_mix * w_min + (1 - lambda_mix) * w_equal

def garlappi_uppal_wang(mean_returns, cov_matrix, risk_free_rate, tau=0.1):
    adjusted_mean = (1 - tau) * mean_returns + tau * risk_free_rate
    return mean_variance_closed_form(adjusted_mean, cov_matrix)

# =============================================================================
# Section 5: Rolling-Window Backtest
# =============================================================================
strategy_names = ["1/N", "MV", "Bayesian Diffuse-Prior", "Bayes-Stein",
                  "Bayesian Data-and-Model", "Min", "VW", "MP",
                  "MV-C", "BS-C", "Min-C", "G-Min-C", "KZ", "EW-Min", "GUW"]

# Dictionaries to store out-of-sample returns and turnover for each strategy.
strategy_returns = {name: [] for name in strategy_names}
strategy_turnover = {name: [] for name in strategy_names}
prev_weights = {name: None for name in strategy_names}

T_out = T_total - L  # Number of out-of-sample periods

for t in range(L, T_total):
    # Use the estimation window: months t-L to t-1
    window_data = simulated_df.iloc[t-L:t]
    est_mu = window_data.mean().values
    est_Sigma = window_data.cov().values
    # Regularize covariance matrix to stabilize inversion.
    est_Sigma = regularize_cov(est_Sigma, epsilon=1e-4)
    
    current_weights = {}
    current_weights["1/N"] = equal_weight_portfolio(N)
    current_weights["MV"] = mean_variance_closed_form(est_mu, est_Sigma)
    diffused_mu = bayesian_diffuse_prior(est_mu, L)
    current_weights["Bayesian Diffuse-Prior"] = mean_variance_closed_form(diffused_mu, est_Sigma)
    bs_mu = bayes_stein_portfolio(est_mu, est_Sigma)
    current_weights["Bayes-Stein"] = mean_variance_closed_form(bs_mu, est_Sigma)
    dm_mu = bayesian_data_and_model_portfolio(est_mu, L)
    current_weights["Bayesian Data-and-Model"] = mean_variance_closed_form(dm_mu, est_Sigma)
    current_weights["Min"] = minimum_variance_portfolio(est_Sigma)
    current_weights["VW"] = equal_weight_portfolio(N)
    current_weights["MP"] = mackinlay_pastor_portfolio(est_mu, est_Sigma)
    current_weights["MV-C"] = mean_variance_optimization_constrained(est_mu, est_Sigma, rf_rate)
    current_weights["BS-C"] = bayes_stein_optimization_constrained(est_mu, est_Sigma, rf_rate)
    current_weights["Min-C"] = minimum_variance_constrained(est_Sigma)
    current_weights["G-Min-C"] = minimum_variance_generalized(est_Sigma, floor=0.05)
    current_weights["KZ"] = kan_zhou_three_fund(est_mu, est_Sigma, rf_rate, gamma=3.0, hedging_coef=-0.1)
    current_weights["EW-Min"] = mixture_minvar_equal(est_Sigma, lambda_mix=0.5)
    current_weights["GUW"] = garlappi_uppal_wang(est_mu, est_Sigma, rf_rate, tau=0.1)
    
    # Out-of-sample returns for month t.
    out_sample_return = simulated_df.iloc[t].values
    
    for name in strategy_names:
        weights = current_weights[name]
        ret = np.dot(weights, out_sample_return)
        strategy_returns[name].append(ret)
        if prev_weights[name] is not None:
            turnover = np.sum(np.abs(weights - prev_weights[name]))
        else:
            turnover = np.nan
        strategy_turnover[name].append(turnover)
        prev_weights[name] = weights

# Create a DataFrame of out-of-sample returns (index = dates from month L onward).
oos_dates = simulated_df.index[L:]
returns_df = pd.DataFrame({name: strategy_returns[name] for name in strategy_names}, index=oos_dates)

# =============================================================================
# Section 6: Compute Performance Metrics for Each Strategy
# =============================================================================
performance_results = {}
for name in strategy_names:
    port_returns = returns_df[name]
    cum_ret = np.expm1(np.log1p(port_returns).sum())
    ann_ret = np.expm1(np.log1p(port_returns).mean() * 12) - 1
    ann_vol = port_returns.std() * np.sqrt(12)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol != 0 else np.nan
    ceq = compute_CEQ(ann_ret, ann_vol, gamma_risk)
    avg_turnover = np.nanmean(strategy_turnover[name])
    performance_results[name] = {
        "Cumulative Return": cum_ret,
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "CEQ": ceq,
        "Turnover": avg_turnover
    }

results_df = pd.DataFrame(performance_results).T
results_df.index.name = "Strategy"
results_df = results_df.reset_index()

print("Simulated Portfolio Strategy Performance (Annualized, based on monthly data):")
print(results_df)

results_df.to_csv("simulated_portfolio_results.csv", index=False)
print("\nResults exported to 'simulated_portfolio_results.csv'")
