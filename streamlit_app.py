import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import TwoSlopeNorm

st.title("Interactive Black-Scholes P&L Heatmaps")

# Sidebar: option parameters
st.sidebar.header("Option Parameters")
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.005)

st.sidebar.header("Purchase Prices")
call_cost = st.sidebar.number_input("Call Purchase Price", min_value=0.0, value=5.0, step=0.1)
put_cost  = st.sidebar.number_input("Put Purchase Price",  min_value=0.0, value=4.0, step=0.1)

st.sidebar.header("Grid Settings")
S_min, S_max = st.sidebar.slider("Spot Price Range", 0.0, 500.0, (80.0, 120.0), step=1.0)
σ_min, σ_max = st.sidebar.slider("Volatility Range", 0.01, 1.0, (0.10, 0.50), step=0.01)
n_S = st.sidebar.slider("Number of Spot Steps",     10, 200, 50, step=5)
n_σ = st.sidebar.slider("Number of Volatility Steps",10, 200, 50, step=5)

st.sidebar.header("Display Options")
show_call = st.sidebar.checkbox("Show Call Heatmap", True)
show_put  = st.sidebar.checkbox("Show Put Heatmap",  True)

# build parameter grids
S_vals     = np.linspace(S_min, S_max, n_S)
sigma_vals = np.linspace(σ_min, σ_max, n_σ)
X, Y = np.meshgrid(sigma_vals, S_vals)

# Black-Scholes pricing
def black_scholes(S, K, T, r, σ, option='call'):
    d1 = (np.log(S/K) + (r + 0.5*σ**2)*T) / (σ * np.sqrt(T))
    d2 = d1 - σ * np.sqrt(T)
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# compute P&L matrices
call_pnl = np.array([
    [black_scholes(S0, K, T, r, σ0, 'call') - call_cost for σ0 in sigma_vals]
    for S0 in S_vals
])
put_pnl = np.array([
    [black_scholes(S0, K, T, r, σ0, 'put')  - put_cost  for σ0 in sigma_vals]
    for S0 in S_vals
])

# determine how many plots to show
plots = []
if show_call: plots.append(("Call P&L", call_pnl))
if show_put:  plots.append(("Put P&L",  put_pnl))

if not plots:
    st.warning("Select at least one heatmap to display.")
else:
    n = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (title, pnl) in zip(axes, plots):
        # per-plot diverging norm
        vmin, vmax = pnl.min(), pnl.max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        pcm = ax.pcolormesh(
            X, Y, pnl,
            cmap='RdYlGn', norm=norm,
            shading='auto',
            edgecolors='white', linewidth=0.2
        )
        ax.set_title(title)
        ax.set_xlabel("Volatility σ")
        ax.set_ylabel("Underlying Price S")
        ax.set_xticks(sigma_vals[::max(1, n_σ//6)])
        ax.set_yticks(S_vals[::max(1, n_S//6)])
        ax.tick_params(axis='x', rotation=45)

        # individual colorbar
        cb = fig.colorbar(pcm, ax=ax, pad=0.02, label="P / L")
        cb.set_ticks(np.linspace(vmin, vmax, 7))

    st.pyplot(fig)
