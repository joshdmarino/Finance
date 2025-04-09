import numpy as np

def MM_quantile(x, q, tol=1e-18, maxit=1e4):
    n = len(x)
    mu_old = np.mean(x)
    for i in range(int(maxit)):
        w = 1/np.abs(x - mu_old)
        mu_new = (sum(w * x) + (2*q - 1)*n)/sum(w)
        if np.isnan(mu_new): return mu_old
        if abs(mu_new - mu_old) < tol: return mu_new
        mu_old = mu_new
    return mu_new

#np.random.seed(42)  # for reproducibility
loss = np.random.normal(loc=-0.05, scale=0.1, size=1000)  # simulated losses (mean -5%, std 10%)

VaR_sim = MM_quantile(loss, 0.99) # 1-day 99% V@R
print(VaR_sim)