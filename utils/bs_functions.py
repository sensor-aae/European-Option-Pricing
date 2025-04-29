import numpy as np
from scipy.stats import norm

# d1 and d2 calculations
def d1(S, K, r, T, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma) - sigma * np.sqrt(T)

# Black-Scholes Price
def black_scholes_price(S, K, r, T, sigma, option_type='call'):
    d_1 = d1(S, K, r, T, sigma)
    d_2 = d2(S, K, r, T, sigma)
    if option_type == 'call':
        return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Greeks
def delta(S, K, r, T, sigma, option_type='call'):
    d_1 = d1(S, K, r, T, sigma)
    if option_type == 'call':
        return norm.cdf(d_1)
    else:
        return norm.cdf(d_1) - 1

def gamma(S, K, r, T, sigma):
    d_1 = d1(S, K, r, T, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))

def vega(S, K, r, T, sigma):
    d_1 = d1(S, K, r, T, sigma)
    return S * norm.pdf(d_1) * np.sqrt(T)
