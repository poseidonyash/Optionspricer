import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy import exp, sqrt, log

# Black-Scholes Model class
class BlackScholes:
    def __init__(self, T, K, S, sigma, r):
        self.T = T  # time to maturity
        self.K = K  # strike price
        self.S = S  # current stock price
        self.sigma = sigma  # volatility
        self.r = r  # risk-free rate

    def d1(self):
        return (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())

    def call_delta(self):
        return norm.cdf(self.d1())

    def put_delta(self):
        return -norm.cdf(-self.d1())

    def call_gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * sqrt(self.T))

    def run(self):
        self.call_price = self.call_price()
        self.put_price = self.put_price()
        self.call_delta = self.call_delta()
        self.put_delta = self.put_delta()
        self.call_gamma = self.call_gamma()

# Page configuration
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    body {
        color: white;
        background-color: #1E1E1E;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: left;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4CAF50;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .card {
        background-color: #2A2A2A;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown("<h1 class='main-header'>📊 Option Prices</h1>", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("<h2 style='color: #4CAF50;'>Model Parameters</h2>", unsafe_allow_html=True)
    
    current_price = st.number_input("Current Asset Price ($)", value=100.0, step=1.0)
    strike = st.number_input("Strike Price ($)", value=95.0, step=1.0)
    time_to_maturity = st.slider("Time to Maturity (Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    interest_rate = st.slider("Risk-Free Interest Rate", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    calculate_btn = st.button('Calculate Option Prices')

# Main content area
if calculate_btn or 'bs_model' not in st.session_state:
    # Calculate prices
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    bs_model.run()
    st.session_state.bs_model = bs_model

# Display option prices
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='card' style='background-color: #4CAF50;'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Call Option Price</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>${st.session_state.bs_model.call_price:.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card' style='background-color: #B07D7D;'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Put Option Price</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>${st.session_state.bs_model.put_price:.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Greeks
st.markdown("<h2 class='sub-header'>🔢 Option Greeks</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Call Delta</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>{st.session_state.bs_model.call_delta:.4f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Put Delta</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>{st.session_state.bs_model.put_delta:.4f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Gamma</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>{st.session_state.bs_model.call_gamma:.4f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Created by: | <a href='https://www.linkedin.com/in/yashprajapati23/' target='_blank' style='color: #4CAF50;'>Yash</a></p>", unsafe_allow_html=True)
