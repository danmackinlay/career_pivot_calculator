# steamlit_app.py — Streamlit app for EV of an AI-safety pivot (stopped Poisson model)
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="Pivot EV Calculator", layout="wide")

# ---------- Model ----------
def ev_exact(p, delta_u, r, ell, T, c):
    """
    Expected value over horizon T (k$ donation-equivalent) as a function of per-app success prob p.
    Stopped Poisson approx: offer rate = r*p, runway = ell. Vectorized in p.
    """
    p = np.asarray(p, dtype=float)
    lamL = r * p * ell
    q = 1.0 - np.exp(-lamL)
    out = np.empty_like(p, dtype=float)
    nz = p > 0
    out[~nz] = -c * ell  # limit as p->0
    if np.any(nz):
        pnz = p[nz]
        out[nz] = q[nz] * (delta_u * T - (delta_u + c) / (r * pnz)) + delta_u * ell * np.exp(-lamL[nz])
    return out

def break_even_p(delta_u, r, ell, T, c, p_lo=1e-6, p_hi=0.5):
    f = lambda p: float(ev_exact(np.array([p]), delta_u, r, ell, T, c))
    f_lo, f_hi = f(p_lo), f(p_hi)
    cap = 0.9
    while np.sign(f_lo) == np.sign(f_hi) and p_hi < cap:
        p_hi = min(cap, p_hi * 1.8)
        f_hi = f(p_hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return None, None
    p_star = brentq(f, p_lo, p_hi)
    q_star = 1.0 - np.exp(-r * p_star * ell)
    return p_star, q_star

# ---------- UI ----------
st.title("Expected Value of an AI-Safety Pivot")
st.caption("Stopped-Poisson approximation (offer rate = r·p) with a capped runway ℓ.")

with st.sidebar:
    st.header("Parameters")
    r = st.number_input("Opportunities per year (r)", min_value=1.0, max_value=200.0, value=24.0, step=1.0)
    ell = st.number_input("Runway (years, ℓ)", min_value=0.05, max_value=2.0, value=0.5, step=0.05, format="%.2f")
    T = st.number_input("Horizon (years, T)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
    c = st.number_input("Runway cost rate (k$/y, c)", min_value=0.0, max_value=200.0, value=25.0, step=1.0)
    delta_u_text = st.text_input("Δu values (k$/y), comma-separated", value="10,22,35")
    p_max = st.slider("Max per-application p shown", min_value=0.02, max_value=0.20, value=0.05, step=0.01)
    grid_pts = st.select_slider("Resolution", options=[200, 400, 800, 1200], value=600)

# Parse Δu list safely
try:
    delta_u_list = [float(s.strip()) for s in delta_u_text.split(",") if s.strip()]
    if not delta_u_list:
        raise ValueError
except Exception:
    st.error("Provide at least one valid Δu value (e.g., 10,22,35).")
    st.stop()

# ---------- Compute ----------
p_grid = np.linspace(0.0, p_max, int(grid_pts))
fig = go.Figure()

for du in delta_u_list:
    ev_vals = ev_exact(p_grid, du, r, ell, T, c)
    # Break-even marker
    p_star, q_star = break_even_p(du, r, ell, T, c)
    name = f"Δu={du:.0f}k/y"
    if p_star is not None:
        name += f"  (p*≈{100*p_star:.2f}%, q*≈{100*q_star:.1f}%)"
        if 0.0 <= p_star <= p_max:
            fig.add_trace(go.Scatter(
                x=[p_star], y=[0.0],
                mode="markers",
                marker=dict(size=8),
                name=f"break-even Δu={du:.0f}",
                hovertemplate=f"Δu={du:.0f}k/y<br>p*={100*p_star:.2f}%<br>q*={100*q_star:.1f}%<extra></extra>"
            ))
    fig.add_trace(go.Scatter(
        x=p_grid, y=ev_vals, mode="lines", name=name,
        hovertemplate="p=%{x:.3f}  (=%{x:.1%})<br>ΔEV=%{y:.2f} k$<extra></extra>"
    ))

# Zero line
fig.add_hline(y=0.0, line_dash="dash")

fig.update_layout(
    xaxis_title="Per-application success probability p",
    yaxis_title="ΔEV over horizon T (k$ donation-equivalent)",
    title=f"EV vs p (r={r:.1f}/y, ℓ={ell:.2f}y, T={T:.1f}y, c={c:.1f}k$/y)",
    legend_title="Scenarios",
    margin=dict(l=40, r=20, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Quick table for a focal Δu ----------
st.subheader("Anchor points")
focal_du = st.selectbox("Focal Δu for table:", options=delta_u_list, index=min(1, len(delta_u_list)-1))
cols = st.columns(3)
for col, pct in zip(cols, [0.01, 0.02, 0.03]):
    with col:
        val = float(ev_exact(np.array([pct]), focal_du, r, ell, T, c))
        st.metric(label=f"ΔEV at p={int(100*pct)}%", value=f"{val:+.2f} k$")

p_star, q_star = break_even_p(focal_du, r, ell, T, c)
if p_star is not None:
    st.info(f"Break-even for Δu={focal_du:.0f}k/y → p* ≈ **{100*p_star:.2f}%** per app; q* over runway ≈ **{100*q_star:.1f}%**.")
else:
    st.warning("No break-even found in search range (EV stays ≥0 or ≤0). Try adjusting parameters or extend the p range.")
