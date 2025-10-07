# streamlit_app.py — EV of an career pivot versus probability of success with log-scale p ∈ [p_min, 1]
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Pivot EV Calculator", layout="wide")


# =========================
# Math helpers (as before)
# =========================
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
    S = lam + rho
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

# ---- Fréchet tail bits (α>1) ----
def s_frechet(alpha, mu_imp):
    if alpha <= 1.0:
        raise ValueError("α must be > 1 for finite mean.")
    return mu_imp / math.gamma(1.0 - 1.0 / alpha)


def C_N_frechet(N, alpha):
    j = np.arange(1, N + 1, dtype=np.float64)
    lg = np.vectorize(math.lgamma)
    return float(np.sum(np.exp(lg(j - 1.0 / alpha) - lg(j))))


def B_frechet_asymptotic(K, N, alpha, mu_imp):
    s = s_frechet(alpha, mu_imp)
    C = C_N_frechet(N, alpha)
    K = np.asarray(K, float)
    return s * C * (K ** (1.0 / alpha))


def I_CF_frechet_asymptotic(K, N, alpha, mu_imp):
    s = s_frechet(alpha, mu_imp)
    C = C_N_frechet(N, alpha)
    K = np.asarray(K, float)
    return (s * C / (N * alpha)) * (K ** (1.0 / alpha))


def W_public_from_B(BK, K, L_fail, delta, N):
    failures = np.clip(np.asarray(K, float) - N, a_min=0.0, a_max=None)
    return BK / delta - failures * L_fail


def p_from_K(K, N, beta=1.0):
    K = np.asarray(K, dtype=float)
    K_safe = np.maximum(K, N / max(beta, 1e-12))
    return np.clip(beta * N / K_safe, 1e-12, 1.0)


def find_zero_crossing(x, y):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    idxs = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if len(idxs) > 0:
        i = int(idxs[0])
        if y[i] == y[i + 1]:
            return 0.5 * (x[i] + x[i + 1]), False
        x0 = np.interp(0.0, [y[i + 1], y[i]], [x[i + 1], x[i]])
        return float(x0), False
    if y[0] <= 0:
        return float(x[0]), True
    if y[-1] > 0:
        return float(x[-1]), True
    return np.nan, True


# =========================
# UI
# =========================
# ---------- UI ----------
st.title("Expected Value of a Career Pivot")
st.markdown(r"""
_Originally inspired by [this blog post](https://danmackinlay.com//notebook/ai_safety_career_calibration). Source code at [danmackinlay/career_pivot_calculator](https://github.com/danmackinlay/career_pivot_calculator)._

You want to know whether to stay in your current job or take a sabbatical to pivot into a role that might have higher impact (e.g., AI safety research, policy, or advocacy).
You expect to apply to many such roles over a limited runway (e.g., 6 months), and each application has some small probability p of success. If you succeed, you get a new job with different pay, donations, and impact. If you fail, you return to your baseline job.
How might you calculate that?
By estimating the *expected value* (EV) of the pivot gamble!.

You apply for jobs at a rate of $r$ per year, and your maximum runway is $\ell$ years (e.g., 0.5 for 6 months).
You discount future utility at a continuous rate $\rho$ per year (e.g., 1/3 for ~3 years half-life if the world’s problems seem _urgent_).
Your runway burn rate is $c$ \$/year (donation-equivalent, i.e., including lost donations and impact).
Y‑axis is present value (k\$ donation‑equivalent) of your sabbatical pivot gamble.
""")


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
        "Discount rate ρ (/y)",
        min_value=0.005,
        max_value=2.0,
        value=1.0 / 3.0,
        step=0.01,
        format="%.3f",
    )
    c = st.number_input(
        "Runway burn c (k$/y)", min_value=0.0, max_value=2000.0, value=50.0, step=1.0
    )

    st.divider()
    st.header("Utility model")
    alpha = st.number_input(
        "Weight on (impact+donations) vs personal $ (α)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    st.subheader("Baseline (stay)")
    w0 = st.number_input(
        "w₀ (k$/y)", min_value=0.0, max_value=3000.0, value=180.0, step=5.0
    )
    d0 = st.number_input(
        "d₀ (k$/y)", min_value=0.0, max_value=3000.0, value=18.0, step=1.0
    )
    i0 = st.number_input(
        "i₀ (k$/y, donation-equiv)",
        min_value=0.0,
        max_value=3000.0,
        value=0.0,
        step=5.0,
    )

    st.subheader("Target role (success)")
    w1 = st.number_input(
        "w₁ (k$/y)", min_value=0.0, max_value=3000.0, value=120.0, step=5.0
    )
    d1 = st.number_input(
        "d₁ (k$/y)", min_value=0.0, max_value=3000.0, value=0.0, step=1.0
    )
    i1 = st.number_input(
        "i₁ (k$/y, donation-equiv)",
        min_value=0.0,
        max_value=3000.0,
        value=100.0,
        step=5.0,
    )

    st.divider()
    st.header("Impact distribution of candidates")
    mu_imp = st.number_input(
        "Mean per-hire impact μ (k$/y)",
        min_value=1.0,
        max_value=1000.0,
        value=22.0,
        step=1.0,
    )
    alpha_tail = st.number_input(
        "Fréchet shape α (>1)",
        min_value=1.05,
        max_value=5.0,
        value=2.0,
        step=0.05,
        format="%.2f",
    )
    N = st.number_input("# roles N", min_value=5, max_value=2000, value=24, step=1)
    epsilon = st.number_input(
        "Externalities ε (k$/y)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0
    )
    use_same_delta = st.checkbox(
        "Use same discount for public ledger (δ = ρ)", value=True
    )
    if use_same_delta:
        delta = rho
    else:
        delta = st.number_input(
            "Public discount δ (/y)",
            min_value=0.005,
            max_value=2.0,
            value=1.0 / 3.0,
            step=0.01,
        )

    st.caption("Large-K asymptotics for Fréchet; for K ≲ 100, interpret cautiously.")

    st.divider()
    st.header("Domain / mapping")
    p_min = st.select_slider(
        "Minimum p to display (controls Kmax)",
        options=[1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        format_func=lambda v: f"{v:.0e}",
    )
    points_per_decade = 200
    beta = st.number_input(
        "β in p≈β·N/K (screening efficiency)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

# ============ Private EV (naive) quick view (your original) ============
delta_u_base, u0, u1 = delta_u_from_components(w0, d0, i0, w1, d1, i1, alpha)
m1, m2, m3 = st.columns(3)
m1.metric("u₀ (stay)", f"{u0:.1f} k$/y")
m2.metric("u₁ (success)", f"{u1:.1f} k$/y")
m3.metric("Δu (naive)", f"{delta_u_base:.1f} k$/y")

# Keep your original private EV vs p (optional – collapsible)
with st.expander("Show simple private EV vs p (naive)"):
    num_decades_A = int(np.ceil(np.log10(1.0 / p_min)))
    npts_A = max(60, int(points_per_decade * num_decades_A))
    p_grid_A = np.geomspace(p_min, 1.0, npts_A)
    figA = go.Figure()
    ev_vals_A = ev_discounted(p_grid_A, delta_u_base, r, ell, rho, c)
    figA.add_trace(
        go.Scatter(
            x=p_grid_A,
            y=ev_vals_A,
            mode="lines",
            name=f"Δu={delta_u_base:.1f} k$/y",
            hovertemplate="p=%{x:.6f} (=%{x:.2%})<br>ΔEV=%{y:.2f} k$<extra></extra>",
        )
    )
    figA.add_hline(y=0.0, line_dash="dash")
    tick_exps_A = list(range(int(np.log10(p_min)), 1))
    figA.update_layout(
        xaxis=dict(
            title="Per-application success p (log)",
            type="log",
            range=[np.log10(p_min), 0],
            tickmode="array",
            tickvals=[10**k for k in tick_exps_A],
            ticktext=[f"1e{k}" if k < 0 else "1" for k in tick_exps_A],
        ),
        yaxis_title="ΔEV (PV, k$)",
        title="[A] Private EV (naive)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(figA, use_container_width=True)


st.subheader("Public/private trade-off (K bottom axis, p top axis)")

# Public cost rate during failure
gamma_pub = i0 + d0 + epsilon
L_fail = gamma_pub * (1.0 - np.exp(-delta * ell)) / delta

# Build K domain from p_min via K ≈ β N / p
K_min = max(N, 20)
K_max = int(np.ceil(beta * N / p_min))
num_decades = int(np.ceil(np.log10(K_max / K_min)))
npts = max(120, int(200 * max(1, num_decades)))
K_grid = np.geomspace(K_min, K_max, npts)
p_grid = p_from_K(K_grid, N=N, beta=beta)

# Counterfactual private surplus & EV
FINANCIAL_DELTA_ALPHA = (w1 - w0) + alpha * (d1 - d0)
I_CF_K = I_CF_frechet_asymptotic(K_grid, N, alpha_tail, mu_imp)
Du_CF_K = FINANCIAL_DELTA_ALPHA + alpha * I_CF_K
EV_CF_K = ev_discounted(p_grid, Du_CF_K, r, ell, rho, c)

# Naive private EV, expressed along same K sweep (for comparison)
EV_naive_K = ev_discounted(p_grid, delta_u_base, r, ell, rho, c)

# Public welfare
BK = B_frechet_asymptotic(K_grid, N, alpha_tail, mu_imp)
W_K = W_public_from_B(BK, K_grid, L_fail, delta, N=N)

# Public optimum K* (closed form)
s = s_frechet(alpha_tail, mu_imp)
C = C_N_frechet(N, alpha_tail)
den = alpha_tail * gamma_pub * (1.0 - np.exp(-delta * ell))
K_star = ((s * C) / den) ** (alpha_tail / (alpha_tail - 1.0)) if den > 0 else np.nan
p_at_K_star = float(p_from_K(K_star, N=N, beta=beta)) if np.isfinite(K_star) else np.nan
W_at_K_star = (
    float(
        W_public_from_B(
            B_frechet_asymptotic(K_star, N, alpha_tail, mu_imp),
            K_star,
            L_fail,
            delta,
            N,
        )
    )
    if np.isfinite(K_star)
    else np.nan
)

# Private equilibrium K_eq where EV_CF=0
K_eq, boundary_eq = find_zero_crossing(K_grid, EV_CF_K)
p_at_K_eq = float(p_from_K(K_eq, N=N, beta=beta)) if np.isfinite(K_eq) else np.nan

# Build plot with dual y and dual x (top axis overlays bottom axis with p ticks)
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Private EVs (left y)
fig.add_trace(
    go.Scatter(
        x=K_grid,
        y=EV_naive_K,
        mode="lines",
        name="A: ΔEV (naive)",
        line=dict(color="black"),
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=K_grid,
        y=EV_CF_K,
        mode="lines",
        name="ΔEV_CF (counterfactual)",
        line=dict(dash="dash"),
    ),
    secondary_y=False,
)
fig.add_hline(y=0.0, line_dash="dot")

# Public welfare (right y)
fig.add_trace(
    go.Scatter(x=K_grid, y=W_K, mode="lines", name="W(K) (public PV)"),
    secondary_y=True,
)

# Vertical markers
if np.isfinite(K_star) and K_min <= K_star <= K_max * 1.2:
    fig.add_vline(
        x=K_star,
        line_dash="dot",
        line_color="#888",
        annotation_text=f"K*≈{int(round(K_star))}",
        annotation_position="top right",
    )
if np.isfinite(K_eq) and K_min <= K_eq <= K_max * 1.2:
    fig.add_vline(
        x=K_eq,
        line_dash="dash",
        line_color="#666",
        annotation_text=f"K_eq≈{int(round(K_eq))}",
        annotation_position="bottom right",
    )

# Bottom x-axis = K (log)
fig.update_xaxes(
    type="log",
    title_text="Applicant pool size K (field view)",
    range=[np.log10(K_min), np.log10(K_max)],
)

# Top x-axis overlay showing p ticks positioned at K=βN/p
# Choose nice p ticks within [p_min, 1]
p_tick_vals = np.array([1.0, 0.2, 0.1, 0.05, 0.02, 0.01])
p_tick_vals = p_tick_vals[(p_tick_vals >= p_min) & (p_tick_vals <= 1.0)]
K_for_p_ticks = beta * N / p_tick_vals

fig.update_layout(
    xaxis2=dict(
        overlaying="x",
        side="top",
        tickmode="array",
        tickvals=K_for_p_ticks,
        ticktext=[f"{pt:.0%}" if pt >= 0.01 else f"{pt:.1%}" for pt in p_tick_vals],
        title="Per-application success p (via p≈β·N/K)",
    ),
    yaxis=dict(title="Private EV (PV, k$)"),
    yaxis2=dict(title="Public welfare W(K) (PV, impact-$)", rangemode="tozero"),
    title=f"Unified trade-off (Fréchet α={alpha_tail:.2f}, N={N}, μ={mu_imp:.0f}, β={beta:.2f})",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=50, r=40, t=80, b=50),
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Model details"):
    st.markdown(r"""
- Mapping $p\approx \beta N/K$ couples candidate view and field view.
- **Private (naive):** $ \Delta EV_\rho(p)=\frac{1-e^{-(rp+\rho)\ell}}{rp+\rho}\Big(\frac{\Delta u\, rp}{\rho}-c\Big) $.
- **Counterfactual impact (Fréchet):** $ \mathcal{I}_{CF}(K)\approx \dfrac{s\,C_N(\alpha)}{N\alpha} K^{1/\alpha} $, with $ s=\mu/\Gamma(1-1/\alpha) $,
  $ C_N(\alpha)=\sum_{k=1}^N \Gamma(k-1/\alpha)/\Gamma(k) $.
- **Counterfactual private surplus:** $ \Delta u_{CF}(K)=(w_1-w_0)+\alpha(d_1-d_0)+\alpha\,\mathcal{I}_{CF}(K) $.
- **Public welfare:** $ W(K)=\dfrac{B(K)}{\delta}-\max\{K-N,0\}L_{\text{fail},\delta} $ with
  $ B(K)\approx s\,C_N(\alpha)K^{1/\alpha} $ and $ L_{\text{fail},\delta}=\gamma(1-e^{-\delta\ell})/\delta $, $ \gamma=i_0+d_0+\varepsilon $.
- **Public optimum:** $ K^*=\Big(\dfrac{s\,C_N(\alpha)}{\alpha\,\gamma\,(1-e^{-\delta\ell})}\Big)^{\!\alpha/(\alpha-1)} $.
""")
