import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from numpy import log, sqrt, exp

# Page configuration
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide", page_icon="💹")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; margin-top: 2rem; margin-bottom: 1rem;}
    .card {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-value {font-size: 2rem; font-weight: bold;}
    .metric-label {font-size: 1rem; color: #888;}
</style>
""", unsafe_allow_html=True)

# Black-Scholes Model class
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        call_price = self.current_price * norm.cdf(d1) - self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        put_price = self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2) - self.current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        self.gamma = norm.pdf(d1) / (self.current_price * self.volatility * sqrt(self.time_to_maturity))

        return call_price, put_price

# Sidebar for inputs
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>💹 Black-Scholes Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Input Parameters</p>", unsafe_allow_html=True)
    
    current_price = st.number_input("Current Asset Price ($)", value=100.0, step=1.0)
    strike = st.number_input("Strike Price ($)", value=100.0, step=1.0)
    time_to_maturity = st.slider("Time to Maturity (Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    interest_rate = st.slider("Risk-Free Interest Rate", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

# Main content
st.markdown("<h1 class='main-header'>Black-Scholes Option Pricing Model</h1>", unsafe_allow_html=True)

# Calculate prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display option prices
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='card' style='background-color: #e6f3ff;'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Call Option Price</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>${call_price:.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card' style='background-color: #fff0e6;'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Put Option Price</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>${put_price:.2f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Greeks
st.markdown("<h2 class='sub-header'>Option Greeks</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Call Delta</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>{bs_model.call_delta:.4f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Put Delta</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>{bs_model.put_delta:.4f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='metric-label'>Gamma</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric-value'>{bs_model.gamma:.4f}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Interactive plots
st.markdown("<h2 class='sub-header'>Interactive Option Price Analysis</h2>", unsafe_allow_html=True)

# Parameter ranges for analysis
spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
vol_range = np.linspace(max(0.01, volatility * 0.5), min(1.0, volatility * 1.5), 100)

# Calculate option prices for different spot prices and volatilities
call_prices = np.zeros((len(vol_range), len(spot_range)))
put_prices = np.zeros((len(vol_range), len(spot_range)))

for i, vol in enumerate(vol_range):
    for j, spot in enumerate(spot_range):
        bs_temp = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate)
        call_prices[i, j], put_prices[i, j] = bs_temp.calculate_prices()

# Create 3D surface plots
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                    subplot_titles=("Call Option Price", "Put Option Price"))

fig.add_trace(go.Surface(z=call_prices, x=spot_range, y=vol_range, colorscale='Blues'), row=1, col=1)
fig.add_trace(go.Surface(z=put_prices, x=spot_range, y=vol_range, colorscale='Reds'), row=1, col=2)

fig.update_layout(scene = dict(xaxis_title='Spot Price', yaxis_title='Volatility', zaxis_title='Option Price'),
                  scene2 = dict(xaxis_title='Spot Price', yaxis_title='Volatility', zaxis_title='Option Price'),
                  width=1200, height=600)

st.plotly_chart(fig)

# Sensitivity Analysis
st.markdown("<h2 class='sub-header'>Sensitivity Analysis</h2>", unsafe_allow_html=True)

sensitivity_param = st.selectbox("Select parameter for sensitivity analysis:", 
                                 ["Current Price", "Volatility", "Time to Maturity"])

if sensitivity_param == "Current Price":
    x_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
    call_prices = [BlackScholes(time_to_maturity, strike, x, volatility, interest_rate).calculate_prices()[0] for x in x_range]
    put_prices = [BlackScholes(time_to_maturity, strike, x, volatility, interest_rate).calculate_prices()[1] for x in x_range]
    x_label = "Current Price"
elif sensitivity_param == "Volatility":
    x_range = np.linspace(0.01, 1, 100)
    call_prices = [BlackScholes(time_to_maturity, strike, current_price, x, interest_rate).calculate_prices()[0] for x in x_range]
    put_prices = [BlackScholes(time_to_maturity, strike, current_price, x, interest_rate).calculate_prices()[1] for x in x_range]
    x_label = "Volatility"
else:  # Time to Maturity
    x_range = np.linspace(0.1, 5, 100)
    call_prices = [BlackScholes(x, strike, current_price, volatility, interest_rate).calculate_prices()[0] for x in x_range]
    put_prices = [BlackScholes(x, strike, current_price, volatility, interest_rate).calculate_prices()[1] for x in x_range]
    x_label = "Time to Maturity (Years)"

sensitivity_fig = go.Figure()
sensitivity_fig.add_trace(go.Scatter(x=x_range, y=call_prices, mode='lines', name='Call Option'))
sensitivity_fig.add_trace(go.Scatter(x=x_range, y=put_prices, mode='lines', name='Put Option'))
sensitivity_fig.update_layout(title=f'Option Prices vs {sensitivity_param}',
                              xaxis_title=x_label, yaxis_title='Option Price',
                              legend_title="Option Type", height=500)

st.plotly_chart(sensitivity_fig)

st.markdown("---")
st.markdown("Created by: Your Name | [LinkedIn](your_linkedin_url)")
