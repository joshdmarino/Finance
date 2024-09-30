import requests
from bs4 import BeautifulSoup

def get_bond_ytm(ticker):
    url = f'https://www.marketwatch.com/investing/stock/{ticker}/bonds'  # Placeholder URL for bonds
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Example: Find YTM from a generic table (site structure may vary)
            ytm_element = soup.find('td', text='Yield to Maturity').find_next_sibling('td')
            
            if ytm_element:
                ytm = ytm_element.text.strip()
                return f"The Yield to Maturity for the corporate bond of {ticker.upper()} is {ytm}."
            else:
                return f"Could not find YTM information for {ticker.upper()}."
        else:
            return f"Failed to retrieve data for {ticker.upper()}. Status code: {response.status_code}"
    
    except Exception as e:
        return f"An error occurred: {e}"

# Ask for user input for stock ticker
ticker = input("Please enter the stock ticker: ").strip()
ytm_info = get_bond_ytm(ticker)
print(ytm_info)
