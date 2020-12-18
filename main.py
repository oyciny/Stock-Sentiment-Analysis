from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

# Collect tickers to be tracked from user
def collect_tickers_to_track():
    x = []
    print("Input stock symbols you wish to view seperated by a space.")
    print("Example: AAPL F TSLA NFLX")
    y = input('Tickers to Track: ').split(' ')
    for symbol in y:
        x.append(symbol)
    return x

# Scrape data on tickers
def data_collection(ticker):
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'Stock-Sentiment'})
    res = urlopen(req)
    html = BeautifulSoup(res, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# Parse the data collected
def parse_data_collected(data):
    parsed_tables = []
    for ticker, news_table in data.items():
        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = row.td.text.split(' ')
            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
            parsed_tables.append([ticker, date, time, title])
    return parsed_tables

def create_data_frame(data):
    x = pd.DataFrame(data, columns=['ticker', 'date', 'time', 'title'])
    x['date'] = pd.to_datetime(x.date).dt.date
    return x

def apply_sentiment(data_frame):
    vader = SentimentIntensityAnalyzer()
    f = lambda x: vader.polarity_scores(x)['compound']
    data_frame['sentiment'] = data_frame['title'].apply(f)
    return data_frame

def calc_mean_sentiment(data_frame):
    data_frame = data_frame.groupby(['ticker', 'date']).mean()
    return data_frame

def format_data_frame(data_frame):
    data_frame = data_frame.unstack()
    data_frame = data_frame.xs('sentiment', axis='columns').transpose()
    return data_frame

def create_mean_data_frame(data):
    return format_data_frame(calc_mean_sentiment(apply_sentiment(create_data_frame(parse_data_collected(data)))))


if __name__ == '__main__':
    tickers = collect_tickers_to_track()
    for ticker in tickers:
        data_collection(ticker)
    df = create_mean_data_frame(news_tables)

    # Plot the data
    df.plot(kind='bar')
    plt.xlabel('Date')
    plt.ylabel('Daily Sentiment Value')
    plt.title('Stock News Sentiment Analysis')
    plt.show()