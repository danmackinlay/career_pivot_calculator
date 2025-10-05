import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Career Pivot Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Career Pivot Calculator")
st.markdown("""
Welcome to the Career Pivot Calculator! This tool helps you analyze and visualize 
career transition decisions using data-driven insights.
""")

# Sidebar for user inputs
st.sidebar.header("Career Parameters")

# Example parameters for career pivot calculation
current_salary = st.sidebar.number_input(
    "Current Annual Salary ($)",
    min_value=0,
    max_value=1000000,
    value=75000,
    step=5000
)

expected_salary = st.sidebar.number_input(
    "Expected Salary After Pivot ($)",
    min_value=0,
    max_value=1000000,
    value=65000,
    step=5000
)

years_experience = st.sidebar.slider(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=5
)

transition_months = st.sidebar.slider(
    "Expected Transition Time (months)",
    min_value=1,
    max_value=24,
    value=6
)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Salary Comparison")
    
    # Create a comparison bar chart
    fig_comparison = go.Figure(data=[
        go.Bar(name='Current', x=['Salary'], y=[current_salary], marker_color='indianred'),
        go.Bar(name='After Pivot', x=['Salary'], y=[expected_salary], marker_color='lightsalmon')
    ])
    fig_comparison.update_layout(
        title="Salary Comparison",
        yaxis_title="Annual Salary ($)",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

with col2:
    st.subheader("Career Growth Projection")
    
    # Generate sample growth projection data
    years = np.arange(0, 11)
    current_trajectory = current_salary * (1.03 ** years)
    pivot_trajectory = expected_salary * (1.05 ** years)
    
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=years, y=current_trajectory,
        mode='lines+markers',
        name='Current Path',
        line=dict(color='indianred', width=2)
    ))
    fig_growth.add_trace(go.Scatter(
        x=years, y=pivot_trajectory,
        mode='lines+markers',
        name='Pivot Path',
        line=dict(color='lightsalmon', width=2)
    ))
    fig_growth.update_layout(
        title="10-Year Growth Projection",
        xaxis_title="Years from Now",
        yaxis_title="Annual Salary ($)",
        height=400
    )
    st.plotly_chart(fig_growth, use_container_width=True)

# Statistical Analysis Section
st.subheader("Statistical Analysis")

col3, col4, col5 = st.columns(3)

with col3:
    # Calculate break-even point
    if expected_salary > current_salary:
        salary_diff = expected_salary - current_salary
        st.metric("Immediate Salary Change", f"${salary_diff:,.0f}", delta=f"{(salary_diff/current_salary)*100:.1f}%")
    else:
        salary_diff = current_salary - expected_salary
        st.metric("Immediate Salary Change", f"-${salary_diff:,.0f}", delta=f"-{(salary_diff/current_salary)*100:.1f}%")

with col4:
    # Calculate opportunity cost
    opportunity_cost = (current_salary / 12) * transition_months
    st.metric("Transition Opportunity Cost", f"${opportunity_cost:,.0f}")

with col5:
    # Calculate break-even time
    if expected_salary > current_salary:
        breakeven = "Immediate gain"
    elif expected_salary == current_salary:
        breakeven = "No difference"
    else:
        # Simplified break-even assuming 3% vs 5% growth
        years_to_breakeven = np.where(pivot_trajectory > current_trajectory)[0]
        if len(years_to_breakeven) > 0:
            breakeven = f"{years_to_breakeven[0]} years"
        else:
            breakeven = ">10 years"
    st.metric("Break-even Point", breakeven)

# Normal distribution visualization for uncertainty
st.subheader("Risk Assessment")
st.markdown("Visualizing salary outcome uncertainty using normal distribution")

# Generate normal distribution data
salary_range = np.linspace(
    expected_salary * 0.7,
    expected_salary * 1.3,
    100
)
std_dev = expected_salary * 0.15
pdf = stats.norm.pdf(salary_range, expected_salary, std_dev)

fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=salary_range, y=pdf,
    fill='tozeroy',
    name='Probability Distribution',
    line=dict(color='royalblue')
))
fig_risk.add_vline(x=expected_salary, line_dash="dash", line_color="red", 
                   annotation_text="Expected Salary")
fig_risk.update_layout(
    title="Salary Outcome Probability Distribution",
    xaxis_title="Potential Salary ($)",
    yaxis_title="Probability Density",
    height=400
)
st.plotly_chart(fig_risk, use_container_width=True)

# Additional insights
st.subheader("Key Insights")
st.info(f"""
- **Experience Level**: {years_experience} years
- **Transition Duration**: {transition_months} months
- **Growth Rate Assumption**: Current path 3% annual, Pivot path 5% annual
- **Risk Level**: Moderate (±15% standard deviation on expected salary)
""")

# Footer
st.markdown("---")
st.markdown("*Note: This calculator provides estimates based on simplified assumptions. Actual career outcomes may vary.*")
