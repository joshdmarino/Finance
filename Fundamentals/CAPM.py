#ER = Rf + b(ERm-Rf)

from turtle import clear
import requests
from bs4 import BeautifulSoup

result = requests.get("https://www.cnbc.com/quotes/US10Y")
src = result.content
soup = BeautifulSoup(src, 'lxml')
Rf = float((soup.find(lambda tag: tag.name == 'span' and tag.get('class') == ['QuoteStrip-lastPrice'])).text[0:4])

ERm = 10.0

# Plan on web scraping data to regress beta automatically 
b = 1.0

ER = Rf + b*(ERm - Rf)

print(ER)