# Career Pivot Calculator 📊

A Streamlit web application for analyzing and visualizing career transition decisions using data-driven insights.

## Features

- **Salary Comparison**: Compare current and expected post-pivot salaries
- **Growth Projection**: Visualize 10-year career trajectory for both paths
- **Statistical Analysis**: Calculate break-even points and opportunity costs
- **Risk Assessment**: Probability distribution visualization for salary outcomes
- **Interactive Dashboard**: Adjust parameters in real-time to explore different scenarios

## Live Demo

🚀 Try the app on Streamlit Community Cloud: [Coming Soon]

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/danmackinlay/career_pivot_calculator.git
cd career_pivot_calculator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Dependencies

- **streamlit**: Web application framework
- **scipy**: Scientific computing and statistical distributions
- **plotly**: Interactive visualizations
- **numpy**: Numerical computing
- **pandas**: Data manipulation

See `requirements.txt` for specific versions.

## Deployment

This app is configured for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Fork this repository
2. Sign in to Streamlit Community Cloud
3. Create a new app and point it to your forked repository
4. The app will automatically deploy using `streamlit_app.py` and `requirements.txt`

## Usage

1. Adjust the parameters in the sidebar:
   - Current annual salary
   - Expected salary after career pivot
   - Years of experience
   - Expected transition time

2. View the visualizations:
   - Salary comparison bar chart
   - 10-year growth projection
   - Statistical metrics (salary change, opportunity cost, break-even point)
   - Risk assessment probability distribution

3. Use the insights to make informed career decisions

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.