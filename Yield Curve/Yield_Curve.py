import os
import plotly.graph_objects as go
from fredapi import Fred

# Replace with your actual FRED API key
fred = Fred(api_key="d03e34e57ff80ba13d471e037e0d8e93")

# Treasury yield series from FRED
series_ids = {
    "1M": "DTB4WK",   # 4-Week Bill
    "3M": "DTB3",     # 3-Month Bill
    "6M": "DTB6",     # 6-Month Bill
    "1Y": "DGS1",     # 1-Year Note
    "2Y": "DGS2",     # 2-Year Note
    "3Y": "DGS3",     # 3-Year Note
    "5Y": "DGS5",     # 5-Year Note
    "7Y": "DGS7",     # 7-Year Note
    "10Y": "DGS10",   # 10-Year Note
    "20Y": "DGS20",   # 20-Year Bond
    "30Y": "DGS30"    # 30-Year Bond
}

bonds = list(series_ids.keys())
bondsy = []

# Fetch latest bond yields
for bond, series in series_ids.items():
    try:
        yield_value = fred.get_series_latest_release(series)
        bondsy.append(yield_value)
        print(f"Yield for {bond}: {yield_value:.2f}%")
    except Exception as e:
        print(f"Failed to fetch {bond}: {e}")
        bondsy.append(None)  # Add None if data is missing

# Determine yield curve status
status = "Yield Curve"
if bondsy[8] and bondsy[4] and bondsy[8] < bondsy[4]:  # 10Y < 2Y means inversion
    status = "Inverted Yield Curve"

# Create interactive plot with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=bonds, y=bondsy, mode='lines+markers', name='US Treasury Yields'))

# Update layout
fig.update_layout(
    title=status,
    xaxis_title="Bond Maturity",
    yaxis_title="Interest Rate (%)",
    template="plotly_dark"
)

# Show interactive graph
fig.show()

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output file path
output_file = os.path.join(script_dir, "yield_curve.html")

# Save plot as HTML file in the same directory as the script
fig.write_html(output_file)

print(f"HTML report saved to: {output_file}")
