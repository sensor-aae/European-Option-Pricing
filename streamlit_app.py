import streamlit as st
from utils.bs_functions import black_scholes_price, delta, gamma, vega

st.set_page_config(page_title="European Option Pricing", layout="centered")

st.title("European Option Pricing")

st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Underlying price (S)", min_value=0.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike price (K)", min_value=0.0, value=100.0, step=1.0)
r = st.sidebar.number_input("Risk-free interest rate (r)", value=0.05, step=0.01, format="%.4f")
T = st.sidebar.number_input("Time to maturity (T in years)", min_value=0.0, value=1.0, step=0.1, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.0, value=0.2, step=0.01, format="%.4f")
option_type = st.sidebar.selectbox("Option type", ("call", "put"))

# Calculate results
price = black_scholes_price(S, K, r, T, sigma, option_type)
delta_val = delta(S, K, r, T, sigma, option_type)
gamma_val = gamma(S, K, r, T, sigma)
vega_val = vega(S, K, r, T, sigma)

st.subheader("Results")
st.write(f"{option_type.capitalize()} option price: {price:.4f}")
st.write(f"Delta: {delta_val:.4f}")
st.write(f"Gamma: {gamma_val:.4f}")
st.write(f"Vega: {vega_val:.4f}")
