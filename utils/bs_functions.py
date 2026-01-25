import numpy as np
from scipy.stats import norm

# d1 and d2 calculations
def d1(S, K, r, T, sigma):
    """Compute the d1 term in the Black–Scholes formula.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price of the option.
    r : float
        Risk‑free interest rate.
    T : float
        Time to maturity (in years).
    sigma : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The d1 value used in the Black–Scholes formula.
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))



def d2(S, K, r, T, sigma):
    """Compute the d2 term in the Black–Scholes formula.

    d2 is derived from d1 by subtracting sigma * sqrt(T).
    See Black–Scholes equation for details.

    Parameters
    ----------
    S, K, r, T, sigma : float
        Same as for d1().

    Returns
    -------
    float
        The d2 value used in the Black–Scholes formula.
    """
    return d1(S, K, r, T, sigma) - sigma * np.sqrt(T)



# Black‑Scholes Price

def black_scholes_price(S, K, r, T, sigma, option_type='call'):
    """Calculate the price of a European option using the Black–Scholes formula.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price of the option.
    r : float
        Risk‑free interest rate.
    T : float
        Time to maturity (in years).
    sigma : float
        Volatility of the underlying asset.
    option_type : {'call', 'put'}, optional
        Type of the option to price. Default is 'call'.

    Returns
    -------
    float
        The price of the option.
    """
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
    """Calculate the Delta of a European option.

    Delta measures the sensitivity of the option price to changes in the underlying asset price.

    Parameters are the same as for black_scholes_price().

    Returns
    -------
    float
        The Delta of the option.
    """
    d_1 = d1(S, K, r, T, sigma)
    if option_type == 'call':
        return norm.cdf(d_1)
    else:
        return norm.cdf(d_1) - 1



def gamma(S, K, r, T, sigma):
    """Calculate the Gamma of a European option.

    Gamma measures the rate of change of Delta with respect to the underlying asset price.

    Returns
    -------
    float
        The Gamma of the option.
    """
    d_1 = d1(S, K, r, T, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))



def vega(S, K, r, T, sigma):
    """Calculate the Vega of a European option.

    Vega measures the sensitivity of the option price to changes in volatility.

    Returns
    -------
    float
        The Vega of the option.
    """
    d_1 = d1(S, K, r, T, sigma)
    return S * norm.pdf(d_1) * np.sqrt(T)
