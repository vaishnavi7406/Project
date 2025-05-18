import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import requests
from bs4 import BeautifulSoup
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Cache data fetching functions to improve performance
@st.cache_data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="3y")  # Fetch last 3 years of data
    return data

@st.cache_data
def fetch_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    industry = info.get('industry', 'N/A')
    volume = info.get('regularMarketVolume', 'N/A')
    beta = info.get('beta', 'N/A')
    return current_price, industry, volume, beta

# Scrape real-time news from Yahoo Finance
def fetch_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    news_items = []
    for item in soup.find_all("li", class_="js-stream-content Pos(r)"):
        title = item.find("h3").text if item.find("h3") else "No title available"
        link = item.find("a")["href"] if item.find("a") else "#"
        if not link.startswith("http"):
            link = "https://finance.yahoo.com" + link
        news_items.append({"title": title, "link": link})
    
    return news_items[:5]  # Return top 5 news articles

# Fallback random financial news and insights
def fetch_random_news():
    random_news = [
        {"title": "Stock Market Hits All-Time High", "link": "https://finance.yahoo.com"},
        {"title": "Tech Stocks Rally Amid Earnings Season", "link": "https://finance.yahoo.com"},
        {"title": "Federal Reserve Hints at Rate Cuts", "link": "https://finance.yahoo.com"},
        {"title": "Global Markets React to Geopolitical Tensions", "link": "https://finance.yahoo.com"},
        {"title": "Energy Sector Surges as Oil Prices Climb", "link": "https://finance.yahoo.com"},
    ]
    return random.sample(random_news, min(5, len(random_news)))  # Return random 5 news items

# Risk analysis
def calculate_risk(data, ticker):
    volatility = data['Close'].std()
    beta = fetch_stock_info(ticker)[3]  # Fetch beta value
    return volatility, beta

# Sentiment analysis
def sentiment_analysis(current_price, predicted_price):
    if predicted_price > current_price:
        return {
            "recommendation": "Buy 🚀",
            "comment": "The stock is expected to rise! A great time to invest.",
            "color": "green",
            "positive": 70,  # Positive sentiment percentage
            "neutral": 20,   # Neutral sentiment percentage
            "negative": 10   # Negative sentiment percentage
        }
    elif predicted_price < current_price:
        return {
            "recommendation": "Sell 🔴",
            "comment": "The stock is expected to drop. Consider selling.",
            "color": "red",
            "positive": 10,  # Positive sentiment percentage
            "neutral": 20,   # Neutral sentiment percentage
            "negative": 70   # Negative sentiment percentage
        }
    else:
        return {
            "recommendation": "Hold 🟠",
            "comment": "The stock is expected to remain stable. Hold your position.",
            "color": "orange",
            "positive": 20,  # Positive sentiment percentage
            "neutral": 70,   # Neutral sentiment percentage
            "negative": 10   # Negative sentiment percentage
        }

# Generate sentiment pie chart
def generate_sentiment_pie_chart(sentiment):
    labels = ["Positive", "Neutral", "Negative"]
    values = [sentiment["positive"], sentiment["neutral"], sentiment["negative"]]
    colors = ["#87CEEB", "#1E3A8A","#FFB6C1"]  # Sky Blue → Light Pink → Dark Blue  
  # Soft pink to coral gradient


    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # Creates a donut chart
        marker=dict(colors=colors),
        textinfo="percent+label",
        hoverinfo="label+percent",
        pull=[0.1, 0, 0]  # Pull the "Positive" slice for animation
    )])

    fig.update_layout(
        title="Sentiment Analysis",
        showlegend=False,
        annotations=[dict(text="Score", x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    return fig

# Predict stock prices using selected model
def predict_stock_prices(data, days, model_type):
    if data.empty:
        raise ValueError("No data available for the given ticker.")
    
    data['Date'] = data.index
    data['Date'] = data['Date'].map(datetime.datetime.toordinal)
    
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values.reshape(-1, 1)
    
    if model_type == "Polynomial Regression":
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]
        future_dates_poly = poly.transform(np.array(future_dates_ordinal).reshape(-1, 1))

        predictions = model.predict(future_dates_poly).flatten()

    elif model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)

        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]

        predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1)).flatten()

    elif model_type == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(5, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=days)
        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]

    elif model_type == "LSTM":
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']].values)

        # Prepare data for LSTM
        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 60
        X_train, y_train = create_dataset(scaled_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)  # Reduced epochs for speed

        # Predict future prices
        last_60_days = scaled_data[-time_step:]
        predictions = []
        for _ in range(days):
            X_test = last_60_days.reshape(1, time_step, 1)
            pred_price = model.predict(X_test, verbose=0)
            predictions.append(pred_price[0][0])
            last_60_days = np.append(last_60_days[1:], pred_price)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        last_date = data.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]

    return future_dates, predictions

# Generate historical stock price graph
def generate_graph(data, chart_type):
    if chart_type == "Line Chart":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data["Close"], 
            mode='lines',
            name="Stock Price",
            line=dict(color='royalblue', width=2)
        ))
    elif chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )])
    elif chart_type == "OHLC":
        fig = go.Figure(data=[go.Ohlc(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        )])
    elif chart_type == "Bar Chart":
        fig = go.Figure(data=[go.Bar(
            x=data.index,
            y=data['Close'],
            name="Bar Chart"
        )])

    fig.update_layout(
        title="Stock Price Over Time",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        template="plotly_dark",  # Default dark theme
        hovermode="x unified"
    )
    
    return fig

# Generate predicted stock price graph
def generate_prediction_graph(future_dates, predictions, chart_type):
    if chart_type == "Line Chart":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=predictions, 
            mode='lines+markers',
            name="Predicted Price",
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=8, color='red', symbol='circle-open')
        ))
    elif chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=future_dates,
            open=predictions,
            high=predictions,
            low=predictions,
            close=predictions,
            name="Candlestick"
        )])
    elif chart_type == "OHLC":
        fig = go.Figure(data=[go.Ohlc(
            x=future_dates,
            open=predictions,
            high=predictions,
            low=predictions,
            close=predictions,
            name="OHLC"
        )])
    elif chart_type == "Bar Chart":
        fig = go.Figure(data=[go.Bar(
            x=future_dates,
            y=predictions,
            name="Bar Chart"
        )])

    fig.update_layout(
        title="Predicted Stock Prices for Upcoming Days",
        xaxis_title="Date",
        yaxis_title="Predicted Close Price (USD)",
        template="plotly_dark",  # Default dark theme
        hovermode="x unified"
    )
    
    return fig

# Generate combined graph (historical + predicted)
def generate_combined_graph(data, future_dates, predictions, chart_type):
    fig = go.Figure()

    if chart_type == "Line Chart":
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode='lines',
            name="Historical Price",
            line=dict(color='royalblue', width=2)
        ))

        # Add predicted data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name="Predicted Price",
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=8, color='red', symbol='circle-open')
        ))

    elif chart_type == "Candlestick":
        # Combine historical and predicted data for candlestick
        combined_dates = list(data.index) + future_dates
        combined_prices = list(data["Close"]) + list(predictions)
        fig.add_trace(go.Candlestick(
            x=combined_dates,
            open=combined_prices,
            high=combined_prices,
            low=combined_prices,
            close=combined_prices,
            name="Combined Prices"
        ))

    elif chart_type == "OHLC":
        # Combine historical and predicted data for OHLC
        combined_dates = list(data.index) + future_dates
        combined_prices = list(data["Close"]) + list(predictions)
        fig.add_trace(go.Ohlc(
            x=combined_dates,
            open=combined_prices,
            high=combined_prices,
            low=combined_prices,
            close=combined_prices,
            name="Combined Prices"
        ))

    elif chart_type == "Bar Chart":
        # Combine historical and predicted data for bar chart
        combined_dates = list(data.index) + future_dates
        combined_prices = list(data["Close"]) + list(predictions)
        fig.add_trace(go.Bar(
            x=combined_dates,
            y=combined_prices,
            name="Combined Prices"
        ))

    fig.update_layout(
        title="Combined Historical and Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",  # Default dark theme
        hovermode="x unified"
    )
    
    return fig

# Risk Score Calculation
def calculate_risk_score(volatility):
    if volatility < 0.1:
        return "Low"
    elif volatility < 0.3:
        return "Moderate"
    else:
        return "High"

# Stability Score Calculation
def calculate_stability_score(beta):
    if beta < 1:
        return "Safe Stock (Low Beta)"
    else:
        return "Volatile Stock (High Beta)"

# Fetch Insider Trading Data
def fetch_insider_trading(ticker):
    stock = yf.Ticker(ticker)
    insider = stock.insider_transactions
    if insider is not None and not insider.empty:
        return insider.head(5)  # Return top 5 insider transactions
    else:
        return None

# Fetch Sector-Wise Performance
def fetch_sector_performance():
    sectors = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Materials": "XLB",
        "Industrials": "XLI",
        "Communication Services": "XLC",
        "Consumer Staples": "XLP"
    }
    sector_performance = {}
    for sector, ticker in sectors.items():
        data = yf.Ticker(ticker).history(period="1mo")
        if not data.empty:
            performance = (data["Close"][-1] - data["Close"][0]) / data["Close"][0] * 100
            sector_performance[sector] = round(performance, 2)
    return sector_performance

# Compare Stocks Side by Side
def compare_stocks(tickers):
    comparison_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = stock.history(period="3y")
        if not data.empty:
            comparison_data.append({
                "Ticker": ticker,
                "Current Price": info.get('currentPrice', 'N/A'),
                "Industry": info.get('industry', 'N/A'),
                "Volume": info.get('regularMarketVolume', 'N/A'),
                "Beta": info.get('beta', 'N/A'),
                "3Y Growth (%)": (data["Close"][-1] - data["Close"][0]) / data["Close"][0] * 100
            })
    return pd.DataFrame(comparison_data)

# Financial Health Check for Stocks
def financial_health_check(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    profitability = info.get('profitMargins', 0) * 100  # Profitability (%)
    debt_levels = info.get('debtToEquity', 0)  # Debt to Equity Ratio
    cash_flow = info.get('operatingCashflow', 0)  # Operating Cash Flow
    roe = info.get('returnOnEquity', 0) * 100  # Return on Equity (%)

    # Calculate Health Score (1-10)
    health_score = (
        (profitability / 20) +  # Max 5 points
        (10 - min(debt_levels, 10)) +  # Max 5 points
        (cash_flow / 1e9) +  # Max 5 points
        (roe / 20)  # Max 5 points
    )
    health_score = min(max(health_score, 1), 10)  # Ensure score is between 1 and 10

    return {
        "Profitability 🏦": f"{profitability:.2f}%",
        "Debt Levels ⚖️": f"{debt_levels:.2f}",
        "Cash Flow 💵": f"${cash_flow / 1e9:.2f}B",
        "Return on Equity (ROE) 📊": f"{roe:.2f}%",
        "Health Score": f"{health_score:.1f}/10"
    }

# Historical Market Insights & Fun Facts
def fetch_market_facts():
    facts = [
        "On this day in 2008, Lehman Brothers collapsed!",
        "Apple's IPO price was just $22 in 1980!",
        "The Dow Jones Industrial Average was first calculated in 1896.",
        "The first stock exchange was established in Amsterdam in 1602.",
        "The 1929 stock market crash led to the Great Depression."
    ]
    return random.choice(facts)

# Currency Converter with Country/Region
def manual_currency_converter(amount, from_currency, to_currency):
    # Define manual conversion rates (as of a specific date)
    conversion_rates = {
        "USD": 1.0,    # United States Dollar
        "EUR": 0.85,   # Euro (European Union)
        "GBP": 0.73,   # British Pound Sterling (United Kingdom)
        "JPY": 110.0,  # Japanese Yen (Japan)
        "INR": 87.15,  # Indian Rupee (India)
        "AUD": 1.35,   # Australian Dollar (Australia)
        "CAD": 1.25,   # Canadian Dollar (Canada)
        "CHF": 0.92,   # Swiss Franc (Switzerland)
        "CNY": 6.45,   # Chinese Yuan (China)
        "SEK": 8.65,   # Swedish Krona (Sweden)
        "NZD": 1.45,   # New Zealand Dollar (New Zealand)
    }


    if from_currency not in conversion_rates or to_currency not in conversion_rates:
        return None

    # Convert to USD first
    amount_in_usd = amount / conversion_rates[from_currency]
    # Convert to target currency
    converted_amount = amount_in_usd * conversion_rates[to_currency]

    return converted_amount

# Streamlit App
def main():
    st.set_page_config(page_title="TradeSense", layout="wide")
    
    # Custom CSS for animations, spacing, and glassmorphism
    st.markdown("""
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .fade-in {
                animation: fadeIn 1s ease-in-out;
            }
            @keyframes slideInLeft {
                from { opacity: 0; transform: translateX(-50px); }
                to { opacity: 1; transform: translateX(0); }
            }
            .slide-in-left {
                animation: slideInLeft 1s ease-in-out;
            }
            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(50px); }
                to { opacity: 1; transform: translateX(0); }
            }
            .slide-in-right {
                animation: slideInRight 1s ease-in-out;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .glass-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
            }
            .metric-title {
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
            }
            .stButton>button {
                background: linear-gradient(45deg, #6a11cb, #2575fc);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .stDataFrame {
                margin-bottom: 20px;
            }
            .stPlotlyChart {
                margin-bottom: 20px;
            }
            .stMarkdown h2 {
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .center-text {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    # Title and description
    st.markdown("""
        <div class="center-text">
            <h1>💹TradeSense 🤖</h1>
             <h3>Trade With Sense | Grow With Confidence </h3>
            <p style="font-size: 18px;">
                🔍 Analyze Any Stock</p>
        <p>Enter a stock symbol and date range to begin analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Input Parameters⚙️")
        ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL) ", value="AAPL").upper()
        days = st.number_input("Days to Predict (1-365)", min_value=1, max_value=365, value=30)
        model_type = st.selectbox("Select Prediction Model", ["Polynomial Regression", "Linear Regression", "ARIMA", "LSTM"])
        chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Candlestick", "OHLC", "Bar Chart"])
        real_time_update = st.checkbox("Enable Real-time Data Updates🚀")

        if st.button("Predict🎯"):
            if not ticker:
                st.error("Please enter a valid ticker symbol.")
            else:
                st.session_state.predict_clicked = True

        # Add "Don't know about ticker?" sentence and "Find Tickers" button
        st.markdown("""
            <div style="margin-top: 10px;">
                <p>Don't know about ticker❓ </p>
                <a href="https://vaishnavi7406.github.io/TickerSense/" target="_blank">
                    <button style="background: linear-gradient(45deg, #6a11cb, #2575fc); color: white; border: none; border-radius: 10px; padding: 10px 20px; font-size: 16px; transition: transform 0.3s ease, box-shadow 0.3s ease;">
                        Find Tickers 🔍
                    </button>
                </a>
            </div>
        """, unsafe_allow_html=True)

        # Currency Converter
        st.header("Currency Converter 💱")
        amount = st.number_input("Amount", min_value=0.01, value=1.0)
        
        # Currency options with country/region
        currency_options = [
            "USD (United States Dollar)",
            "EUR (Euro - European Union)",
            "GBP (British Pound Sterling - United Kingdom)",
            "JPY (Japanese Yen - Japan)",
            "INR (Indian Rupee - India)",
            "AUD (Australian Dollar - Australia)",
            "CAD (Canadian Dollar - Canada)",
            "CHF (Swiss Franc - Switzerland)",
            "CNY (Chinese Yuan - China)",
            "SEK (Swedish Krona - Sweden)",
            "NZD (New Zealand Dollar - New Zealand)"
        ]
        
        # Set default values for From and To currencies
        from_currency = st.selectbox("From Currency", currency_options, index=0)  # Default: USD
        to_currency = st.selectbox("To Currency", currency_options, index=4)     # Default: INR
        
        # Extract currency code from the selected option
        from_currency_code = from_currency.split(" ")[0]
        to_currency_code = to_currency.split(" ")[0]
        
        if st.button("Convert🔄"):
            converted_amount = manual_currency_converter(amount, from_currency_code, to_currency_code)
            if converted_amount is not None:
                st.success(f"Converted Amount: {converted_amount:.2f} {to_currency_code}")
            else:
                st.error("Currency conversion failed. Please check the currency codes.")

        # Trading Simulator Link
        st.markdown("""
            <div style="margin-top: 20px;">
                <h3>Don't know about trading🤷‍♂️ Learn here📚!</h3>
                <a href="https://traderiserrai.streamlit.app/" target="_blank">
                    <button style="background: linear-gradient(45deg, #6a11cb, #2575fc); color: white; border: none; border-radius: 10px; padding: 10px 20px; font-size: 16px; transition: transform 0.3s ease, box-shadow 0.3s ease;">
                        Go to Trading Simulator▶️
                    </button>
                </a>
            </div>
        """, unsafe_allow_html=True)

    # Display results only if "Predict" is clicked
    if st.session_state.predict_clicked:
        try:
            # Fetch and process data
            data = fetch_stock_data(ticker)
            if data.empty:
                st.error("No data available for the given ticker. Please check the ticker symbol.")
            else:
                future_dates, predictions = predict_stock_prices(data, days, model_type)
                company_name = yf.Ticker(ticker).info.get("shortName", ticker)
                current_price, industry, volume, beta = fetch_stock_info(ticker)
                predicted_price = predictions[0]  # First predicted price
                sentiment = sentiment_analysis(current_price, predicted_price)

                # Display results with animations
                st.markdown("""
                    <div class="fade-in">
                        <h2 style="text-align: center;">{} ({}) - Prediction Results 📊</h2>
                    </div>
                """.format(company_name, ticker), unsafe_allow_html=True)

                # Info Boxes with Glassmorphism Effect
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <div class="metric-title">💵 Current Price</div>
                            <div class="metric-value">${:.2f}</div>
                        </div>
                    """.format(current_price), unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <div class="metric-title">🏭 Industry</div>
                            <div class="metric-value">{}</div>
                        </div>
                    """.format(industry), unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                        <div class="glass-card fade-in">
                            <div class="metric-title">📈 Volume</div>
                            <div class="metric-value">{:,}</div>
                        </div>
                    """.format(volume), unsafe_allow_html=True)

                # Risk Analysis
                st.subheader("📊 Risk Analysis")
                volatility, beta = calculate_risk(data, ticker)  # Pass ticker to calculate_risk
                risk_score = calculate_risk_score(volatility)
                stability_score = calculate_stability_score(beta)
                st.markdown(f"""
                    <div class="glass-card fade-in">
                        <div class="metric-title">📉 Volatility</div>
                        <div class="metric-value">{volatility:.2f}</div>
                    </div>
                    <div class="glass-card fade-in">
                        <div class="metric-title">📊 Beta</div>
                        <div class="metric-value">{beta}</div>
                    </div>
                    <div class="glass-card fade-in">
                        <div class="metric-title">📊 Risk Score</div>
                        <div class="metric-value">{risk_score}</div>
                    </div>
                    <div class="glass-card fade-in">
                        <div class="metric-title">📊 Stability Score</div>
                        <div class="metric-value">{stability_score}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Insider Trading Data
                st.subheader("📊 Insider Trading Activity")
                insider_trading = fetch_insider_trading(ticker)
                if insider_trading is not None:
                    st.dataframe(insider_trading)
                else:
                    st.warning("No insider trading data available for this ticker.")

                # Sector-Wise Performance
                st.subheader("📊 Sector-Wise Performance")
                sector_performance = fetch_sector_performance()
                if sector_performance:
                    st.write(pd.DataFrame.from_dict(sector_performance, orient="index", columns=["Performance (%)"]))
                else:
                    st.warning("Unable to fetch sector-wise performance data.")

                # Recent Stock Data
                st.subheader("📅 Recent Stock Data")
                st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5))

                # Historical Stock Price Chart
                st.subheader("📈 Historical Stock Price Chart")
                st.plotly_chart(generate_graph(data, chart_type), use_container_width=True)

                # Predicted Stock Prices
                st.subheader("🔮 Predicted Stock Prices")
                st.dataframe(pd.DataFrame({
                    "Date": [date.date() for date in future_dates],
                    "Predicted Price": [f"${price:.2f}" for price in predictions]
                }))

                # Prediction Graph
                st.subheader("📊 Prediction Graph")
                st.plotly_chart(generate_prediction_graph(future_dates, predictions, chart_type), use_container_width=True)

                # Combined Historical and Predicted Graph
                st.subheader("📊 Combined Historical and Predicted Prices")
                st.plotly_chart(generate_combined_graph(data, future_dates, predictions, chart_type), use_container_width=True)

                # Sentiment Analysis
                st.subheader("🎯 Recommendation")
                st.markdown(f"""
                    <div style="background-color: {sentiment['color']}; padding: 20px; border-radius: 15px; text-align: center;">
                        <h2>{sentiment['recommendation']}</h2>
                        <p>{sentiment['comment']}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Sentiment Score Pie Chart
                st.subheader("📊 Sentiment Score")
                st.plotly_chart(generate_sentiment_pie_chart(sentiment), use_container_width=True)

                # Real-Time News Section (Only if enabled)
                if real_time_update:
                    st.subheader("📰 Real-Time News & Insights")
                    news = fetch_news(ticker)
                    if not news:  # If no news is found, fetch random news
                        news = fetch_random_news()
                        st.warning("No specific news found for this ticker. Here are some general financial insights:")
                    
                    for article in news:
                        st.markdown(f"""
                            <div class="glass-card fade-in">
                                <h4>{article['title']}</h4>
                                <p><a href="{article['link']}" target="_blank">Read more</a></p>
                            </div>
                        """, unsafe_allow_html=True)

                # Compare Stocks Side by Side
                st.subheader("📊 Compare Stocks Side by Side")
                tickers_to_compare = st.multiselect("Select stocks to compare", [ "A","ZYXI"])
                if tickers_to_compare:
                    comparison_data = compare_stocks(tickers_to_compare)
                    st.dataframe(comparison_data)

                # Financial Health Check for Stocks
                st.subheader("📊 Financial Health Check")
                health_check = financial_health_check(ticker)
                st.write(pd.DataFrame.from_dict(health_check, orient="index", columns=["Value"]))

                # Historical Market Insights & Fun Facts
                st.subheader("📜 Did You Know?")
                st.markdown(f"""
                    <div class="glass-card fade-in">
                        <h4>{fetch_market_facts()}</h4>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
