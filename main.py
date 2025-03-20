import re
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.linear_model import LinearRegression


# -------------------------------
# Caching Data Functions
# -------------------------------
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, period="1y"):
    """
    Retrieve historical stock data for the given ticker using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            st.error(f"No data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error retrieving data for {ticker}: {e}")
        return pd.DataFrame()


# -------------------------------
# KPI Data Retrieval
# -------------------------------
@st.cache_data(show_spinner=False)
def get_stock_kpi(ticker):
    """
    Retrieve KPI metrics (latest price, previous close, open) for a given ticker.
    """
    data = get_stock_data(ticker, period="5d")
    if data is not None and not data.empty:
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) >= 2 else latest
        return {
            "Latest Price": f"${latest['Close']:.2f}",
            "Previous Close": f"${previous['Close']:.2f}",
            "Open": f"${latest['Open']:.2f}",
        }
    return {"Latest Price": "N/A", "Previous Close": "N/A", "Open": "N/A"}


# -------------------------------
# Stock Prediction Function
# -------------------------------
def predict_next_close(data):
    """
    Use a simple linear regression on the historical closing prices
    to predict the next day’s closing price.
    """
    if data is None or data.empty or len(data) < 10:
        return None

    data = data.reset_index(drop=True).dropna(subset=["Close"])
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    next_index = np.array([[len(data)]])
    prediction = model.predict(next_index)[0]
    return prediction


# -------------------------------
# NLP Based Ticker Extraction
# -------------------------------
def extract_ticker(user_text):
    """
    Extract a stock ticker from the user search text using simple keyword matching.
    """
    mapping = {
        "apple": "AAPL",
        "aapl": "AAPL",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "googl": "GOOGL",
        "microsoft": "MSFT",
        "msft": "MSFT",
        "amazon": "AMZN",
        "amzn": "AMZN",
        "tesla": "TSLA",
        "tsla": "TSLA",
        "meta": "META",
        "facebook": "META",
        "netflix": "NFLX",
        "nvidia": "NVDA",
        "intel": "INTC",
        "amd": "AMD",
        "cisco": "CSCO",
        "ibm": "IBM"
    }

    text_lower = user_text.lower()
    for key, ticker in mapping.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            return ticker
    return None


# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Prediction & Analysis Application")

# Comprehensive description
st.markdown(
    """
    Introducing our comprehensive **Stock Analytics Platform** that seamlessly integrates live market data, KPI metrics, historical charts, and a prediction model. With a user-friendly, centered search box, simply type in a query like "What is the price of Apple?" to instantly **access live data, key metrics, and a dynamic chart** that displays historical trends alongside a **next‑day closing price prediction.**

Additionally, our intuitive dashboard features **KPI cards** for renowned brands such as Coca‑Cola, Nike, and McDonald’s, paired with a robust grid of 12 top market stocks arranged in a visually engaging layout. Powered by the **yfinance API, a linear regression model for predictions**, and an interactive Streamlit interface, this application is designed to empower you with clear, actionable insights to make **informed investment decisions**.
    """
)

# -------------------------------
# NLP Search Box (Centered)
# -------------------------------
with st.container():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    user_query = st.text_input("Enter your stock query (e.g., 'What is the price of Apple?')", "", key="nlp_search")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Dynamic Large Chart on User Query (Now Popping Right Below the Search Box)
# -------------------------------
if user_query:
    ticker = extract_ticker(user_query)
    if ticker:
        st.markdown("---")
        st.header(f"Stock Analysis for {ticker}")
        data = get_stock_data(ticker)
        if data is not None and not data.empty:
            latest_price = data["Close"].iloc[-1]
            st.write(f"**Latest Closing Price for {ticker}:** ${latest_price:.2f}")
            prediction = predict_next_close(data)
            if prediction:
                st.write(f"**Predicted Next Day Closing Price:** ${prediction:.2f}")
            else:
                st.error("Prediction could not be made due to insufficient data.")

            st.subheader("Historical Data with Next Day Prediction")
            data_plot = data[["Close"]].copy().reset_index(drop=True)
            data_plot.index = pd.date_range(end=datetime.date.today(), periods=len(data_plot), freq="B")
            future_date = data_plot.index[-1] + pd.tseries.offsets.BDay(1)
            data_plot.loc[future_date] = prediction
            st.line_chart(data_plot)
        else:
            st.error(f"Could not retrieve data for ticker {ticker}.")
    else:
        st.error(
            "Could not extract a stock ticker from your query. Try including a company name like 'Apple', 'Google', 'Microsoft', etc.")

# -------------------------------
# KPI Section for Famous Brands
# -------------------------------
st.subheader("Key Performance Indicators (KPIs) for Famous Brands")
famous_brands_kpis = {
    "Coca‑Cola": "KO",
    "Nike": "NKE",
    "McDonald's": "MCD"
}
kpi_cols = st.columns(len(famous_brands_kpis))
for col, (brand, ticker) in zip(kpi_cols, famous_brands_kpis.items()):
    kpi_data = get_stock_kpi(ticker)
    with col:
        st.markdown(f"### {brand} ({ticker})")
        st.metric(label="Latest Price", value=kpi_data.get("Latest Price", "N/A"))
        st.metric(label="Previous Close", value=kpi_data.get("Previous Close", "N/A"))
        st.metric(label="Open", value=kpi_data.get("Open", "N/A"))

# -------------------------------
# Fixed 12 Stock Charts in 3x4 Grid
# -------------------------------
st.subheader("Live Stock Charts for Top 12 Stocks")
top_12_stocks = {
    "Apple": "AAPL",
    "Google (Alphabet)": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta (Facebook)": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA",
    "Intel": "INTC",
    "AMD": "AMD",
    "Cisco": "CSCO",
    "IBM": "IBM"
}

stock_names = list(top_12_stocks.keys())
stock_tickers = list(top_12_stocks.values())
num_stocks = len(stock_names)

for row in range(0, num_stocks, 3):
    cols = st.columns(3)
    for i in range(3):
        idx = row + i
        if idx < num_stocks:
            company = stock_names[idx]
            ticker = stock_tickers[idx]
            data = get_stock_data(ticker)
            with cols[i]:
                st.markdown(f"**{company} ({ticker})**")
                if data is not None and not data.empty:
                    st.line_chart(data["Close"])
                else:
                    st.warning(f"Data for {ticker} is unavailable.")
