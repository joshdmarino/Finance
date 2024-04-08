from turtle import clear
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

bonds = ["1M","3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"]
stocks =["AAPL"]
bondsy = []

for i in range(len(bonds)):
    result = requests.get("https://www.cnbc.com/quotes/US"+bonds[i])
    src = result.content
    soup = BeautifulSoup(src, 'lxml')
    data = soup.find_all(lambda tag: tag.name == 'span' and tag.get('class') == ['QuoteStrip-lastPrice'])
    for element in data:
        #print (bonds[i] + " UST: " + element.text)
        bondsy.append(float(element.text[0:4]))

status = "Yield Curve"
if(bonds[8]<bonds[4]):
    status = "Inverted Yield Curve"

plt.plot(bonds,bondsy)
plt.title(f'{status}')
plt.xlabel("Length")
plt.ylabel("Interest Rate")
plt.show()

