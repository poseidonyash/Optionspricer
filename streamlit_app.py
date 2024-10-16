import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from numpy import exp, sqrt, log

# Black-Scholes Model class (as provided)
class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def run(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

# Page configuration
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide", page_icon="💹")

# Function to create a fancy background
def add_bg_from_base64(base64_string):
    return f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_string}");
        background-size: cover;
    }}
    </style>
    """

# Fancy background (you can replace this with any base64 encoded image)
background_image = """
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==
"""
st.markdown(add_bg_from_base64(background_image), unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 2rem;
        font-weight: bold;
        color: #43A047;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;  /* Center text in all cards */
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100px;  /* Set height to make boxes uniform */
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-top: auto;
        margin-bottom: auto;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-top: auto;
        margin-bottom: auto;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown("<h1 class='main-header'>🌟 Black-Scholes Option Pricing Model 🌟</h1>", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>📊 Model Parameters</h2>", unsafe_allow_html=True)
    
    current_price = st.number_input("Current Asset Price ($)", value=100.0, step=1.0)
    strike = st.number_input("Strike Price ($)", value=100.0, step=1.0)
    time_to_maturity = st.slider("Time to Maturity (Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    interest_rate = st.slider("Risk-Free Interest Rate", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    calculate_btn = st.button('Calculate Option Prices')

# Main content area
if calculate_btn:
    # Calculate prices
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    bs_model.run()

    # Display option prices
    st.markdown("<h2 class='sub-header'>📈 Option Prices</h2>", unsafe_allow_html=True)  
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card' style='background-color: rgba(144, 238, 144, 0.6);'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Call Option Price</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${bs_model.call_price:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card' style='background-color: rgba(255, 182, 193, 0.6);'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Put Option Price</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${bs_model.put_price:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<h2 class='sub-header'>🔢 Option Greeks</h2>", unsafe_allow_html=True)
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
        st.markdown(f"<p class='metric-value'>{bs_model.call_gamma:.4f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive plots
    st.markdown("<h2 class='sub-header'>🎨 Interactive Option Price Analysis</h2>", unsafe_allow_html=True)

    # Parameter ranges for analysis
    spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 50)
    vol_range = np.linspace(max(0.01, volatility * 0.5), min(1.0, volatility * 1.5), 50)

    # Calculate option prices for different spot prices and volatilities
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate)
            bs_temp.run()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price

    # Create 3D surface plots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                        subplot_titles=("Call Option Price", "Put Option Price"))

    fig.add_trace(go.Surface(z=call_prices, x=spot_range, y=vol_range, colorscale='Viridis'), row=1, col=1)
    fig.add_trace(go.Surface(z=put_prices, x=spot_range, y=vol_range, colorscale='Plasma'), row=1, col=2)

    fig.update_layout(scene = dict(xaxis_title='Spot Price', yaxis_title='Volatility', zaxis_title='Option Price'),
                      scene2 = dict(xaxis_title='Spot Price', yaxis_title='Volatility', zaxis_title='Option Price'),
                      width=1200, height=600)

    st.plotly_chart(fig)

else:
    st.info("👈 Adjust the parameters in the sidebar and click 'Calculate Option Prices' to see the results!")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Created by: | <a href='https://www.linkedin.com/in/yashprajapati23/' target='_blank'>Yash</a></p>", unsafe_allow_html=True)
