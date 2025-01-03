import os
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Define bond maturities
bonds = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
bondsy = []

# Scrape bond yield data
for bond in bonds:
    result = requests.get(f"https://www.cnbc.com/quotes/US{bond}")
    src = result.content
    soup = BeautifulSoup(src, 'lxml')
    data = soup.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['QuoteStrip-lastPrice'])
    for element in data:
        bondsy.append(float(element.text[0:4]))

# Determine yield curve status
status = "Yield Curve"
if bondsy[8] < bondsy[4]:
    status = "Inverted Yield Curve"

# Create interactive plot with Plotly
fig = go.Figure()

# Add line plot
fig.add_trace(go.Scatter(x=bonds, y=bondsy, mode='lines+markers', name='US Treasury Yields'))

# Update layout
fig.update_layout(
    title=status,
    xaxis_title="Bond Maturity",
    yaxis_title="Interest Rate (%)",
    template="plotly_dark"
)

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output file path
output_file = os.path.join(script_dir, "yield_curve.html")

# Save plot as HTML file in the same directory as the script
fig.write_html(output_file)

print(f"HTML report saved to: {output_file}")
