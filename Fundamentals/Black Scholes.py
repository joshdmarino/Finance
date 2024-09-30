import math

def black_scholes(S, K, T, r, sigma, option='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    if option == 'call':
        return S * math.norm.cdf(d1) - K * math.exp(-r * T) * math.norm.cdf(d2)
    if option == 'put':
        return K * math.exp(-r * T) * math.norm.cdf(-d2) - S * math.norm.cdf(-d1)

S = 100
K = 105
T = 0.5
r = 0.05
sigma = 0.2

call_price = black_scholes(S, K, T, r, sigma, option='call')
put_price = black_scholes(S, K, T, r, sigma, option='put')

print(f"Call price: {call_price:.2f}")
print(f"Put price: {put_price:.2f}")