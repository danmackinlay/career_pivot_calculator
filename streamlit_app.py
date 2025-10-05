# streamlit_app.py — EV of an AI-safety pivot (discounted, stopped Poisson) with log-scale p ∈ [p_min, 1]
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Pivot EV Calculator (Discounted)", layout="wide")


# ---------- Math ----------
def ev_discounted(p, delta_u, r, ell, rho, c):
    """
    Discounted expected value ΔEV_ρ(p) in k$ present value.
    p: per-application success probability (0..1)
    r: opportunities per year
    ell: runway (years)
    rho: discount rate per year (>0)
    c: runway burn rate (k$/year), donation-equivalent
    delta_u: per-year surplus if pivot succeeds (k$/year)
    Formula: ΔEV_ρ(p) = [(1 - exp(-(r*p + rho)*ell)) / (r*p + rho)] * (delta_u*(r*p)/rho - c)
    Vectorized in p.
    """
    p = np.asarray(p, dtype=float)
    lam = r * p
    S = lam + rho  # > 0 since rho > 0
    # Stable scale term using expm1 when S*ell is small
    scale = -np.expm1(-S * ell) / S
    return scale * (delta_u * lam / rho - c)


def p_star_discounted(r, rho, c, delta_u):
    """
    Closed-form break-even p*: p* = (c * rho) / (r * delta_u), if delta_u > 0; else np.nan.
    """
    return np.nan if delta_u <= 0 else (c * rho) / (r * delta_u)


def q_over_runway(p, r, ell):
    """Total success probability over runway ℓ when arrivals ~ Poisson with rate r*p."""
    return 1.0 - np.exp(-r * p * ell)


def delta_u_from_components(w0, d0, i0, w1, d1, i1, alpha):
    """
    Δu = (w1 + α*(i1 + d1)) - (w0 + α*(i0 + d0)), in k$/year
    """
    u0 = w0 + alpha * (i0 + d0)
    u1 = w1 + alpha * (i1 + d1)
    return (u1 - u0), u0, u1


# ---------- UI ----------
st.title("Expected Value of an AI‑Safety Pivot (Discounted)")
st.caption(
    "Stopped‑Poisson offers (rate = r·p), capped runway ℓ, continuous discounting at rate ρ. "
    "Y‑axis is present value (k$ donation‑equivalent)."
)

with st.sidebar:
    st.header("Opportunity process & discounting")
    r = st.number_input(
        "Opportunities per year (r)",
        min_value=0.1,
        max_value=1000.0,
        value=24.0,
        step=1.0,
    )
    ell = st.number_input(
        "Runway (years, ℓ)",
        min_value=0.05,
        max_value=5.0,
        value=0.5,
        step=0.05,
        format="%.2f",
    )
    rho = st.number_input(
        "Discount rate (per year, ρ)",
        min_value=0.005,
        max_value=2.0,
        value=1.0 / 3.0,
        step=0.01,
        format="%.3f",
    )
    c = st.number_input(
        "Runway burn rate (k$/year, c)",
        min_value=0.0,
        max_value=1000.0,
        value=25.0,
        step=1.0,
    )

    st.divider()
    st.header("Utility model (compose Δu from components)")
    alpha = st.number_input(
        "Weight on (impact + donations) vs personal $ (α)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    st.subheader("Baseline (stay)")
    w0 = st.number_input(
        "Baseline wage w₀ (k$/y)",
        min_value=0.0,
        max_value=2000.0,
        value=180.0,
        step=5.0,
    )
    d0 = st.number_input(
        "Baseline donations d₀ (k$/y)",
        min_value=0.0,
        max_value=2000.0,
        value=18.0,
        step=1.0,
    )
    i0 = st.number_input(
        "Baseline impact i₀ (k$/y, donation‑equivalent)",
        min_value=0.0,
        max_value=2000.0,
        value=0.0,
        step=5.0,
    )

    st.subheader("Target role (pivot succeeds)")
    w1 = st.number_input(
        "New wage w₁ (k$/y)", min_value=0.0, max_value=2000.0, value=120.0, step=5.0
    )
    d1 = st.number_input(
        "New donations d₁ (k$/y)", min_value=0.0, max_value=2000.0, value=0.0, step=1.0
    )
    i1 = st.number_input(
        "New impact i₁ (k$/y, donation‑equivalent)",
        min_value=0.0,
        max_value=2000.0,
        value=100.0,
        step=5.0,
    )

    st.divider()
    st.header("Optional scenarios")
    st.caption("Add extra Δu values (k$/y) to plot alongside the Δu from components.")
    delta_u_text = st.text_input(
        "Extra Δu values (comma‑separated, optional)", value=""
    )

    st.divider()
    st.subheader("p‑axis (log)")
    p_min = 10.0**-3  # start at 0.1%
    points_per_decade = st.slider(
        "Resolution (points/decade)", min_value=50, max_value=400, value=200, step=10
    )

# Compose Δu from components
delta_u_base, u0, u1 = delta_u_from_components(w0, d0, i0, w1, d1, i1, alpha)

# Parse any extra Δu scenarios
extra_dus = []
if delta_u_text.strip():
    try:
        extra_dus = [float(s.strip()) for s in delta_u_text.split(",") if s.strip()]
    except Exception:
        st.error(
            "Could not parse extra Δu values. Use a comma‑separated list like: 10,22,35"
        )
        extra_dus = []

# Build the list of scenarios (base first)
scenarios = [(f"Δu (from components) = {delta_u_base:.1f}k/y", delta_u_base)]
for du in extra_dus:
    scenarios.append((f"Δu = {du:.1f}k/y (extra)", du))

# ---------- Top-line metrics ----------
m1, m2, m3 = st.columns(3)
m1.metric("u₀ (stay)", f"{u0:.1f} k$/y")
m2.metric("u₁ (pivot success)", f"{u1:.1f} k$/y")
m3.metric("Δu (u₁ − u₀)", f"{delta_u_base:.1f} k$/y")

# ---------- Compute p-grid ----------
num_decades = int(np.ceil(np.log10(1.0 / p_min)))
npts = max(50, int(points_per_decade * num_decades))
p_grid = np.geomspace(p_min, 1.0, npts)

# ---------- Plot ----------
fig = go.Figure()

for label, du in scenarios:
    ev_vals = ev_discounted(p_grid, du, r, ell, rho, c)
    # Closed-form break-even if Δu>0
    p_star = p_star_discounted(r, rho, c, du)
    name = label
    if np.isfinite(p_star) and (p_min <= p_star <= 1.0):
        q_star = q_over_runway(p_star, r, ell)
        name += f"  (p*≈{100 * p_star:.2f}%, q*≈{100 * q_star:.1f}%)"
        fig.add_trace(
            go.Scatter(
                x=[p_star],
                y=[0.0],
                mode="markers",
                marker=dict(size=8),
                name=f"break-even ({label})",
                hovertemplate=f"{label}<br>p*={100 * p_star:.2f}%<br>"
                f"q*={100 * q_star:.1f}%<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=p_grid,
            y=ev_vals,
            mode="lines",
            name=name,
            hovertemplate="p=%{x:.6f} (=%{x:.2%})<br>ΔEV(PV)=%{y:.2f} k$<extra></extra>",
        )
    )

fig.add_hline(y=0.0, line_dash="dash")

tick_exps = list(range(int(np.log10(p_min)), 1))  # e.g., -3, -2, -1, 0
fig.update_layout(
    xaxis=dict(
        title="Per‑application success probability p (log scale)",
        type="log",
        range=[np.log10(p_min), 0],
        tickmode="array",
        tickvals=[10**k for k in tick_exps],
        ticktext=[f"1e{k}" if k < 0 else "1" for k in tick_exps],
        minor=dict(showgrid=True),
    ),
    yaxis_title="ΔEV (present value, k$ donation‑equivalent)",
    title=f"EV vs p (discounted) — r={r:.1f}/y, ℓ={ell:.2f}y, ρ={rho:.3f}/y, c={c:.1f}k$/y",
    legend_title="Scenarios",
    margin=dict(l=40, r=20, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Anchor points ----------
st.subheader("Anchor points")
focal_idx = st.selectbox(
    "Choose scenario for table:",
    options=list(range(len(scenarios))),
    format_func=lambda i: scenarios[i][0],
    index=0,
)
focal_label, focal_du = scenarios[focal_idx]

cols = st.columns(3)
for col, pct in zip(cols, [0.001, 0.01, 0.05]):  # 0.1%, 1%, 5%
    with col:
        val = float(ev_discounted(np.array([pct]), focal_du, r, ell, rho, c))
        st.metric(label=f"ΔEV at p={pct:.1%}", value=f"{val:+.2f} k$")

p_star = p_star_discounted(r, rho, c, focal_du)
if np.isfinite(p_star):
    q_star = q_over_runway(p_star, r, ell)
    if p_min <= p_star <= 1.0:
        st.info(
            f"Break‑even for **{focal_label}** → p* ≈ **{100 * p_star:.2f}%** per app; "
            f"q* over runway ≈ **{100 * q_star:.1f}%**."
        )
    else:
        st.warning(
            f"Break‑even exists (p*≈{100 * p_star:.2f}%), but lies outside the plotted range "
            f"[{p_min:.3%}, 100%]. Increase p‑range if needed."
        )
else:
    st.warning(
        f"No break‑even for **{focal_label}** (Δu ≤ 0 ⇒ EV stays ≤ 0 for all p). "
        f"Increase Δu, reduce c or ρ, or increase r."
    )

# ---------- Explanation ----------
with st.expander("Model details"):
    st.markdown(
        """
- **Discounted private EV**:
  \n\\[
  \\Delta \\mathrm{EV}_\\rho(p)
  = \\frac{1-e^{-(rp+\\rho)\\ell}}{rp+\\rho}\\Big(\\frac{\\Delta u\\, rp}{\\rho}-c\\Big).
  \\]
- **Break‑even**: if \\(\\Delta u>0\\), \\(p^*=\\tfrac{c\\,\\rho}{r\\,\\Delta u}\\), and \\(q^*=1-e^{-rp^*\\ell}\\).
- **Utility composition**: \\(\\Delta u=(w_1+\\alpha(i_1+d_1))-(w_0+\\alpha(i_0+d_0))\\).
- Units are **k$ donation‑equivalent** per year and present value.
        """
    )
