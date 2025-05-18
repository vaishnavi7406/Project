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
                tickers_to_compare = st.multiselect("Select stocks to compare", [
    "A","AA","AAC","AACG","AACIW","AADI","AAIC","AAIN","AAL","AAMC","AAME","AAN","AAOI","AAON","AAP","AAPL","AAQC","AAT","AATC","AAU","AAWW","AB","ABB","ABBV","ABC","ABCB","ABCL","ABCM","ABEO","ABEV","ABG","ABIO","ABM","ABMD","ABNB","ABOS","ABR","ABSI","ABST","ABT","ABTX","ABUS","ABVC","AC","ACA","ACAB","ACAD","ACAQ","ACAXR","ACB","ACBA","ACC","ACCD","ACCO","ACEL","ACER","ACET","ACEV","ACEVW","ACGL","ACGLN","ACGLO","ACH","ACHC","ACHL","ACHR","ACHV","ACI","ACII","ACIU","ACIW","ACKIT","ACLS","ACLX","ACM","ACMR","ACN","ACNB","ACON","ACOR","ACP","ACQR","ACQRU","ACR","ACRE","ACRS","ACRX","ACST","ACT","ACTD","ACTDW","ACTG","ACU","ACV","ACVA","ACXP","ADAG","ADALW","ADAP","ADBE","ADC","ADCT","ADER","ADES","ADEX","ADGI","ADI","ADIL","ADM","ADMA","ADMP","ADN","ADNT","ADNWW","ADP","ADPT","ADRA","ADRT","ADSE","ADSEW","ADSK","ADT","ADTH","ADTN","ADTX","ADUS","ADV","ADVM","ADX","ADXN","AE","AEAC","AEACW","AEAE","AEAEW","AEE","AEF","AEFC","AEG","AEHAW","AEHL","AEHR","AEI","AEIS","AEL","AEM","AEMD","AENZ","AEO","AEP","AEPPZ","AER","AERC","AERI","AES","AESC","AESE","AEVA","AEY","AEYE","AEZS","AFAQ","AFAR","AFB","AFBI","AFCG","AFG","AFGB","AFGC","AFGD","AFGE","AFIB","AFL","AFMD","AFRI","AFRIW","AFRM","AFT","AFTR","AFYA","AG","AGAC","AGBAR","AGCB","AGCO","AGD","AGE","AGEN","AGFS","AGFY","AGGR","AGI","AGIL","AGILW","AGIO","AGL","AGLE","AGM","AGMH","AGNC","AGNCM","AGNCN","AGNCO","AGNCP","AGO","AGR","AGRI","AGRO","AGRX","AGS","AGTC","AGTI","AGX","AGYS","AHCO","AHG","AHH","AHI","AHPA","AHPI","AHRNW","AHT","AI","AIB","AIC","AIF","AIG","AIH","AIHS","AIKI","AIM","AIMAW","AIMC","AIN","AINC","AINV","AIO","AIP","AIR","AIRC","AIRG","AIRI","AIRS","AIRT","AIRTP","AIT","AIU","AIV","AIZ","AIZN","AJG","AJRD","AJX","AJXA","AKA","AKAM","AKAN","AKBA","AKICU","AKR","AKRO","AKTS","AKTX","AKU","AKUS","AKYA","AL","ALB","ALBO","ALC","ALCC","ALCO","ALDX","ALE","ALEC","ALEX","ALF","ALFIW","ALG","ALGM","ALGN","ALGS","ALGT","ALHC","ALIM","ALIT","ALJJ","ALK","ALKS","ALKT","ALL","ALLE","ALLG","ALLK","ALLO","ALLR","ALLT","ALLY","ALNA","ALNY","ALORW","ALOT","ALPA","ALPN","ALPP","ALR","ALRM","ALRN","ALRS","ALSA","ALSAR","ALSAU","ALSAW","ALSN","ALT","ALTG","ALTO","ALTR","ALTU","ALTUU","ALTUW","ALV","ALVO","ALVR","ALX","ALXO","ALYA","ALZN","AM","AMAL","AMAM","AMAO","AMAOW","AMAT","AMBA","AMBC","AMBO","AMBP","AMC","AMCI","AMCR","AMCX","AMD","AME","AMED","AMEH","AMG","AMGN","AMH","AMK","AMKR","AMLX","AMN","AMNB","AMOT","AMOV","AMP","AMPE","AMPG","AMPH","AMPI","AMPL","AMPS","AMPY","AMR","AMRC","AMRK","AMRN","AMRS","AMRX","AMS","AMSC","AMSF","AMST","AMSWA","AMT","AMTB","AMTD","AMTI","AMTX","AMWD","AMWL","AMX","AMYT","AMZN","AN","ANAB","ANAC","ANDE","ANEB","ANET","ANF","ANGH","ANGHW","ANGI","ANGN","ANGO","ANIK","ANIP","ANIX","ANNX","ANPC","ANSS","ANTE","ANTX","ANVS","ANY","ANZU","ANZUW","AOD","AOGO","AOMR","AON","AORT","AOS","AOSL","AOUT","AP","APA","APAC","APACW","APAM","APCX","APD","APDN","APEI","APEN","APG","APGB","APH","API","APLD","APLE","APLS","APLT","APM","APMIU","APO","APOG","APP","APPF","APPH","APPHW","APPN","APPS","APRE","APRN","APT","APTM","APTO","APTV","APTX","APVO","APWC","APXI","APYX","AQB","AQMS","AQN","AQNA","AQNB","AQNU","AQST","AQUA","AR","ARAV","ARAY","ARBE","ARBEW","ARBK","ARBKL","ARC","ARCB","ARCC","ARCE","ARCH","ARCK","ARCKW","ARCO","ARCT","ARDC","ARDS","ARDX","ARE","AREB","AREC","AREN","ARES","ARGD","ARGO","ARGU","ARGUU","ARGUW","ARGX","ARHS","ARI","ARIS","ARIZW","ARKO","ARKOW","ARKR","ARL","ARLO","ARLP","ARMK","ARMP","ARNC","AROC","AROW","ARQQ","ARQQW","ARQT","ARR","ARRWU","ARRWW","ARRY","ARTE","ARTEW","ARTL","ARTNA","ARTW","ARVL","ARVN","ARW","ARWR","ASA","ASAI","ASAN","ASAQ","ASAX","ASAXU","ASB","ASC","ASCAU","ASCB","ASCBR","ASG","ASGI","ASGN","ASH","ASIX","ASLE","ASLN","ASM","ASMB","ASML","ASND","ASNS","ASO","ASPA","ASPC","ASPCU","ASPCW","ASPN","ASPS","ASPU","ASR","ASRT","ASRV","ASTC","ASTE","ASTL","ASTLW","ASTR","ASTS","ASTSW","ASUR","ASX","ASXC","ASYS","ASZ","ATA","ATAI","ATAQ","ATAX","ATC","ATCO","ATCX","ATEC","ATEN","ATER","ATEX","ATGE","ATHA","ATHE","ATHM","ATHX","ATI","ATIF","ATIP","ATKR","ATLC","ATLCL","ATLCP","ATLO","ATNF","ATNFW","ATNI","ATNM","ATNX","ATO","ATOM","ATOS","ATR","ATRA","ATRC","ATRI","ATRO","ATSG","ATTO","ATUS","ATVC","ATVCU","ATVI","ATXI","ATXS","ATY","AU","AUB","AUBAP","AUBN","AUD","AUDC","AUGX","AUID","AUMN","AUPH","AUR","AURA","AURC","AUROW","AUS","AUST","AUTL","AUTO","AUUD","AUVI","AUY","AVA","AVAC","AVAH","AVAL","AVAN","AVAV","AVB","AVCO","AVCT","AVCTW","AVD","AVDL","AVDX","AVEO","AVGO","AVGOP","AVGR","AVID","AVIR","AVK","AVLR","AVNS","AVNT","AVNW","AVO","AVPT","AVPTW","AVRO","AVT","AVTE","AVTR","AVTX","AVXL","AVY","AVYA","AWF","AWH","AWI","AWK","AWP","AWR","AWRE","AWX","AX","AXAC","AXDX","AXGN","AXL","AXLA","AXNX","AXON","AXP","AXR","AXS","AXSM","AXTA","AXTI","AXU","AY","AYI","AYLA","AYRO","AYTU","AYX","AZ","AZEK","AZN","AZO","AZPN","AZRE","AZTA","AZUL","AZYO","AZZ","B","BA","BABA","BAC","BACA","BAFN","BAH","BAK","BALL","BALY","BAM","BAMH","BAMI","BAMR","BANC","BAND","BANF","BANFP","BANR","BANX","BAOS","BAP","BARK","BASE","BATL","BATRA","BATRK","BAX","BB","BBAI","BBAR","BBBY","BBCP","BBD","BBDC","BBDO","BBGI","BBI","BBIG","BBIO","BBLG","BBLN","BBN","BBQ","BBSI","BBU","BBUC","BBVA","BBW","BBWI","BBY","BC","BCAB","BCAC","BCACU","BCACW","BCAN","BCAT","BCBP","BCC","BCDA","BCDAW","BCE","BCEL","BCH","BCLI","BCML","BCO","BCOR","BCOV","BCOW","BCPC","BCRX","BCS","BCSA","BCSAW","BCSF","BCTX","BCTXW","BCV","BCX","BCYC","BDC","BDJ","BDL","BDN","BDSX","BDTX","BDX","BDXB","BE","BEAM","BEAT","BECN","BEDU","BEEM","BEKE","BELFA","BELFB","BEN","BENE","BENER","BENEW","BEP","BEPC","BEPH","BEPI","BERY","BEST","BFAC","BFAM","BFC","BFH","BFI","BFIIW","BFIN","BFK","BFLY","BFRI","BFRIW","BFS","BFST","BFZ","BG","BGB","BGCP","BGFV","BGH","BGI","BGNE","BGR","BGRY","BGRYW","BGS","BGSF","BGSX","BGT","BGX","BGXX","BGY","BH","BHAC","BHACU","BHAT","BHB","BHC","BHE","BHF","BHFAL","BHFAM","BHFAN","BHFAO","BHFAP","BHG","BHIL","BHK","BHLB","BHP","BHR","BHSE","BHSEW","BHV","BHVN","BIDU","BIG","BIGC","BIGZ","BIIB","BILI","BILL","BIMI","BIO","BIOC","BIOL","BIOR","BIOSW","BIOT","BIOTU","BIOTW","BIOX","BIP","BIPC","BIPH","BIPI","BIRD","BIT","BITF","BIVI","BJ","BJDX","BJRI","BK","BKCC","BKD","BKE","BKEP","BKEPP","BKH","BKI","BKKT","BKN","BKNG","BKR","BKSC","BKSY","BKT","BKTI","BKU","BKYI","BL","BLBD","BLBX","BLCM","BLCO","BLCT","BLD","BLDE","BLDEW","BLDP","BLDR","BLE","BLEU","BLEUU","BLEUW","BLFS","BLFY","BLI","BLIN","BLK","BLKB","BLMN","BLND","BLNG","BLNGW","BLNK","BLNKW","BLPH","BLRX","BLSA","BLTE","BLTS","BLTSW","BLU","BLUA","BLUE","BLW","BLX","BLZE","BMA","BMAC","BMAQ","BMAQR","BMBL","BME","BMEA","BMEZ","BMI","BMO","BMRA","BMRC","BMRN","BMTX","BMY","BNED","BNFT","BNGO","BNL","BNOX","BNR","BNRG","BNS","BNSO","BNTC","BNTX","BNY","BOAC","BOAS","BOC","BODY","BOE","BOH","BOKF","BOLT","BON","BOOM","BOOT","BORR","BOSC","BOTJ","BOWL","BOX","BOXD","BOXL","BP","BPAC","BPMC","BPOP","BPOPM","BPRN","BPT","BPTH","BPTS","BPYPM","BPYPN","BPYPO","BPYPP","BQ","BR","BRAC","BRACR","BRAG","BRBR","BRBS","BRC","BRCC","BRCN","BRDG","BRDS","BREZ","BREZR","BREZW","BRFH","BRFS","BRG","BRID","BRIV","BRIVW","BRKHU","BRKL","BRKR","BRLI","BRLT","BRMK","BRN","BRO","BROG","BROS","BRP","BRPM","BRPMU","BRPMW","BRQS","BRSP","BRT","BRTX","BRW","BRX","BRY","BRZE","BSAC","BSBK","BSBR","BSET","BSFC","BSGA","BSGAR","BSGM","BSIG","BSKY","BSKYW","BSL","BSM","BSMX","BSQR","BSRR","BST","BSTZ","BSVN","BSX","BSY","BTA","BTAI","BTB","BTBD","BTBT","BTCM","BTCS","BTCY","BTG","BTI","BTMD","BTMDW","BTN","BTO","BTOG","BTRS","BTT","BTTR","BTTX","BTU","BTWN","BTWNU","BTWNW","BTX","BTZ","BUD","BUI","BUR","BURL","BUSE","BV","BVH","BVN","BVS","BVXV","BW","BWA","BWAC","BWACW","BWAQR","BWAY","BWB","BWC","BWCAU","BWEN","BWFG","BWG","BWMN","BWMX","BWNB","BWSN","BWV","BWXT","BX","BXC","BXMT","BXMX","BXP","BXRX","BXSL","BY","BYD","BYFC","BYM","BYN","BYND","BYRN","BYSI","BYTS","BYTSW","BZ","BZFD","BZFDW","BZH","BZUN","C","CAAP","CAAS","CABA","CABO","CAC","CACC","CACI","CADE","CADL","CAE","CAF","CAG","CAH","CAJ","CAKE","CAL","CALA","CALB","CALM","CALT","CALX","CAMP","CAMT","CAN","CANF","CANG","CANO","CAPD","CAPL","CAPR","CAR","CARA","CARE","CARG","CARR","CARS","CARV","CASA","CASH","CASI","CASS","CASY","CAT","CATC","CATO","CATY","CB","CBAN","CBAT","CBAY","CBD","CBFV","CBH","CBIO","CBL","CBNK","CBOE","CBRE","CBRG","CBRL","CBSH","CBT","CBTX","CBU","CBZ","CC","CCAP","CCB","CCBG","CCCC","CCCS","CCD","CCEL","CCEP","CCF","CCI","CCJ","CCK","CCL","CCLP","CCM","CCNC","CCNE","CCNEP","CCO","CCOI","CCRD","CCRN","CCS","CCSI","CCU","CCV","CCVI","CCXI","CCZ","CD","CDAK","CDAY","CDE","CDEV","CDLX","CDMO","CDNA","CDNS","CDR","CDRE","CDRO","CDTX","CDW","CDXC","CDXS","CDZI","CDZIP","CE","CEA","CEAD","CEADW","CECE","CEE","CEG","CEI","CEIX","CELC","CELH","CELU","CELZ","CEM","CEMI","CEN","CENN","CENQW","CENT","CENTA","CENX","CEPU","CEQP","CERE","CERS","CERT","CET","CETX","CETXP","CEV","CEVA","CF","CFB","CFBK","CFFE","CFFI","CFFN","CFG","CFIV","CFIVW","CFLT","CFMS","CFR","CFRX","CFSB","CFVI","CFVIW","CG","CGA","CGABL","CGAU","CGBD","CGC","CGEM","CGEN","CGNT","CGNX","CGO","CGRN","CGTX","CHAA","CHCI","CHCO","CHCT","CHD","CHDN","CHE","CHEA","CHEF","CHEK","CHGG","CHH","CHI","CHK","CHKEL","CHKEW","CHKEZ","CHKP","CHMG","CHMI","CHN","CHNG","CHNR","CHPT","CHRA","CHRB","CHRD","CHRS","CHRW","CHS","CHSCL","CHSCM","CHSCN","CHSCO","CHSCP","CHT","CHTR","CHUY","CHW","CHWA","CHWAW","CHWY","CHX","CHY","CI","CIA","CIB","CIDM","CIEN","CIF","CIFR","CIFRW","CIG","CIGI","CIH","CII","CIIGW","CIK","CIM","CINC","CINF","CING","CINT","CIO","CION","CIR","CISO","CITEW","CIVB","CIVI","CIX","CIXX","CIZN","CJJD","CKPT","CKX","CL","CLAQW","CLAR","CLAS","CLAYU","CLB","CLBK","CLBS","CLBT","CLBTW","CLDT","CLDX","CLEU","CLF","CLFD","CLGN","CLH","CLIM","CLIR","CLLS","CLM","CLMT","CLNE","CLNN","CLOV","CLPR","CLPS","CLPT","CLR","CLRB","CLRO","CLS","CLSD","CLSK","CLSN","CLST","CLVR","CLVRW","CLVS","CLVT","CLW","CLWT","CLX","CLXT","CM","CMA","CMAX","CMAXW","CMBM","CMC","CMCA","CMCL","CMCM","CMCO","CMCSA","CMCT","CME","CMG","CMI","CMLS","CMMB","CMP","CMPO","CMPOW","CMPR","CMPS","CMPX","CMRA","CMRAW","CMRE","CMRX","CMS","CMSA","CMSC","CMSD","CMT","CMTG","CMTL","CMU","CNA","CNC","CNCE","CND","CNDB","CNDT","CNET","CNEY","CNF","CNFRL","CNHI","CNI","CNK","CNM","CNMD","CNNB","CNNE","CNO","CNOB","CNOBP","CNP","CNQ","CNR","CNS","CNSL","CNSP","CNTA","CNTB","CNTG","CNTQ","CNTQW","CNTX","CNTY","CNVY","CNX","CNXA","CNXC","CNXN","CO","COCO","COCP","CODA","CODI","CODX","COE","COF","COFS","COGT","COHN","COHU","COIN","COKE","COLB","COLD","COLI","COLIU","COLIW","COLL","COLM","COMM","COMP","COMS","COMSP","COMSW","CONN","CONX","CONXW","COO","COOK","COOL","COOLU","COOP","COP","CORR","CORS","CORT","CORZ","CORZW","COSM","COST","COTY","COUP","COUR","COVA","COVAU","COVAW","COWN","COWNL","CP","CPA","CPAC","CPAR","CPARU","CPARW","CPB","CPE","CPF","CPG","CPHC","CPHI","CPIX","CPK","CPLP","CPNG","CPOP","CPRI","CPRT","CPRX","CPS","CPSH","CPSI","CPSS","CPT","CPTK","CPTN","CPTNW","CPUH","CPZ","CQP","CR","CRAI","CRBP","CRBU","CRC","CRCT","CRDF","CRDL","CRDO","CREC","CREG","CRESW","CRESY","CREX","CRF","CRGE","CRGY","CRH","CRHC","CRI","CRIS","CRK","CRKN","CRL","CRM","CRMD","CRMT","CRNC","CRNT","CRNX","CRON","CROX","CRS","CRSP","CRSR","CRT","CRTD","CRTDW","CRTO","CRTX","CRU","CRUS","CRVL","CRVS","CRWD","CRWS","CRXT","CRXTW","CS","CSAN","CSBR","CSCO","CSCW","CSGP","CSGS","CSII","CSIQ","CSL","CSPI","CSQ","CSR","CSSE","CSSEN","CSSEP","CSTE","CSTL","CSTM","CSTR","CSV","CSWC","CSWI","CSX","CTAQ","CTAS","CTBB","CTBI","CTDD","CTEK","CTG","CTGO","CTHR","CTIB","CTIC","CTKB","CTLP","CTLT","CTMX","CTO","CTOS","CTR","CTRA","CTRE","CTRM","CTRN","CTS","CTSH","CTSO","CTT","CTV","CTVA","CTXR","CTXRW","CTXS","CUBA","CUBE","CUBI","CUE","CUEN","CUK","CULL","CULP","CURI","CURO","CURV","CUTR","CUZ","CVAC","CVBF","CVCO","CVCY","CVE","CVEO","CVET","CVGI","CVGW","CVI","CVII","CVLG","CVLT","CVLY","CVM","CVNA","CVR","CVRX","CVS","CVT","CVV","CVX","CW","CWAN","CWBC","CWBR","CWCO","CWEN","CWH","CWK","CWST","CWT","CX","CXAC","CXDO","CXE","CXH","CXM","CXW","CYAN","CYBE","CYBN","CYBR","CYCC","CYCCP","CYCN","CYD","CYH","CYN","CYRN","CYRX","CYT","CYTH","CYTK","CYTO","CYXT","CZNC","CZOO","CZR","CZWI","D","DAC","DADA","DAIO","DAKT","DAL","DALN","DAN","DAO","DAOO","DAOOU","DAOOW","DAR","DARE","DASH","DATS","DAVA","DAVE","DAVEW","DAWN","DB","DBD","DBGI","DBI","DBL","DBRG","DBTX","DBVT","DBX","DC","DCBO","DCF","DCFC","DCFCW","DCGO","DCGOW","DCI","DCO","DCOM","DCOMP","DCP","DCPH","DCRD","DCRDW","DCT","DCTH","DD","DDD","DDF","DDI","DDL","DDOG","DDS","DDT","DE","DEA","DECA","DECK","DEI","DELL","DEN","DENN","DEO","DESP","DEX","DFFN","DFH","DFIN","DFP","DFS","DG","DGHI","DGICA","DGII","DGLY","DGNU","DGX","DH","DHACW","DHBC","DHBCU","DHC","DHCAU","DHCNI","DHCNL","DHF","DHHC","DHI","DHIL","DHR","DHT","DHX","DHY","DIAX","DIBS","DICE","DIN","DINO","DIOD","DIS","DISA","DISH","DIT","DK","DKL","DKNG","DKS","DLA","DLB","DLCA","DLHC","DLNG","DLO","DLPN","DLR","DLTH","DLTR","DLX","DLY","DM","DMA","DMAC","DMB","DMF","DMLP","DMO","DMRC","DMS","DMTK","DNA","DNAA","DNAB","DNAC","DNAD","DNAY","DNB","DNLI","DNMR","DNN","DNOW","DNP","DNUT","DNZ","DO","DOC","DOCN","DOCS","DOCU","DOGZ","DOLE","DOMA","DOMO","DOOO","DOOR","DORM","DOUG","DOV","DOW","DOX","DOYU","DPG","DPRO","DPSI","DPZ","DQ","DRCT","DRD","DRE","DRH","DRI","DRIO","DRMA","DRMAW","DRQ","DRRX","DRTS","DRTSW","DRTT","DRUG","DRVN","DS","DSAC","DSACU","DSACW","DSEY","DSGN","DSGR","DSGX","DSKE","DSL","DSM","DSP","DSS","DSU","DSWL","DSX","DT","DTB","DTC","DTE","DTEA","DTF","DTG","DTIL","DTM","DTOCW","DTP","DTSS","DTST","DTW","DUK","DUKB","DUNE","DUNEW","DUO","DUOL","DUOT","DV","DVA","DVAX","DVN","DWAC","DWACU","DWACW","DWIN","DWSN","DX","DXC","DXCM","DXF","DXLG","DXPE","DXR","DXYN","DY","DYAI","DYFN","DYN","DYNT","DZSI","E","EA","EAC","EACPW","EAD","EAF","EAI","EAR","EARN","EAST","EAT","EB","EBACU","EBAY","EBC","EBET","EBF","EBIX","EBMT","EBON","EBR","EBS","EBTC","EC","ECAT","ECC","ECCC","ECCW","ECCX","ECF","ECL","ECOM","ECOR","ECPG","ECVT","ED","EDAP","EDBL","EDBLW","EDD","EDF","EDI","EDIT","EDN","EDNC","EDR","EDRY","EDSA","EDTK","EDTX","EDU","EDUC","EE","EEA","EEFT","EEIQ","EEX","EFC","EFL","EFOI","EFR","EFSC","EFSCP","EFT","EFTR","EFX","EGAN","EGBN","EGF","EGHT","EGIO","EGLE","EGLX","EGO","EGP","EGRX","EGY","EH","EHAB","EHC","EHI","EHTH","EIC","EICA","EIG","EIGR","EIM","EIX","EJH","EKSO","EL","ELA","ELAN","ELAT","ELBM","ELC","ELDN","ELEV","ELF","ELMD","ELOX","ELP","ELS","ELSE","ELTK","ELV","ELVT","ELY","ELYM","ELYS","EM","EMAN","EMBC","EMBK","EMBKW","EMCF","EMD","EME","EMF","EMKR","EML","EMLD","EMN","EMO","EMP","EMR","EMWP","EMX","ENB","ENBA","ENCP","ENCPW","ENDP","ENER","ENERW","ENFN","ENG","ENIC","ENJ","ENJY","ENJYW","ENLC","ENLV","ENO","ENOB","ENOV","ENPC","ENPH","ENR","ENS","ENSC","ENSG","ENSV","ENTA","ENTFW","ENTG","ENTX","ENTXW","ENV","ENVA","ENVB","ENVX","ENX","ENZ","EOCW","EOD","EOG","EOI","EOLS","EOS","EOSE","EOSEW","EOT","EP","EPAC","EPAM","EPC","EPD","EPHY","EPHYU","EPHYW","EPIX","EPM","EPR","EPRT","EPSN","EPWR","EPZM","EQ","EQBK","EQC","EQD","EQH","EQHA","EQIX","EQNR","EQOS","EQR","EQRX","EQRXW","EQS","EQT","EQX","ERAS","ERC","ERES","ERESU","ERF","ERH","ERIC","ERIE","ERII","ERJ","ERO","ERYP","ES","ESAB","ESAC","ESCA","ESE","ESEA","ESGR","ESGRO","ESGRP","ESI","ESLT","ESMT","ESNT","ESOA","ESPR","ESQ","ESRT","ESS","ESSA","ESSC","ESSCW","ESTA","ESTC","ESTE","ET","ETAC","ETACW","ETB","ETD","ETG","ETJ","ETN","ETNB","ETO","ETON","ETR","ETRN","ETSY","ETTX","ETV","ETW","ETWO","ETX","ETY","EUCR","EURN","EVA","EVAX","EVBG","EVBN","EVC","EVCM","EVER","EVEX","EVF","EVFM","EVG","EVGN","EVGO","EVGOW","EVGR","EVH","EVI","EVK","EVLO","EVLV","EVM","EVN","EVO","EVOJ","EVOJU","EVOJW","EVOK","EVOP","EVR","EVRG","EVRI","EVT","EVTC","EVTL","EVTV","EVV","EW","EWBC","EWCZ","EWTX","EXAI","EXAS","EXC","EXD","EXEL","EXFY","EXG","EXK","EXLS","EXN","EXP","EXPD","EXPE","EXPI","EXPO","EXPR","EXR","EXTN","EXTR","EYE","EYEN","EYES","EYPT","EZFL","EZGO","EZPW","F","FA","FACA","FACT","FAF","FAM","FAMI","FANG","FANH","FARM","FARO","FAST","FAT","FATBB","FATBP","FATE","FATH","FATP","FAX","FBC","FBHS","FBIO","FBIOP","FBIZ","FBK","FBMS","FBNC","FBP","FBRT","FBRX","FC","FCAP","FCAX","FCBC","FCCO","FCEL","FCF","FCFS","FCN","FCNCA","FCNCO","FCNCP","FCO","FCPT","FCRD","FCRX","FCT","FCUV","FCX","FDBC","FDEU","FDMT","FDP","FDS","FDUS","FDX","FE","FEAM","FEDU","FEI","FELE","FEMY","FEN","FENC","FENG","FEO","FERG","FET","FEXD","FEXDR","FF","FFA","FFBC","FFC","FFHL","FFIC","FFIE","FFIEW","FFIN","FFIV","FFNW","FFWM","FGB","FGBI","FGBIP","FGEN","FGF","FGFPP","FGI","FGIWW","FGMC","FHB","FHI","FHN","FHS","FHTX","FIAC","FIACW","FIBK","FICO","FIF","FIGS","FINM","FINMW","FINS","FINV","FINW","FIS","FISI","FISV","FITB","FITBI","FITBO","FITBP","FIVE","FIVN","FIX","FIXX","FIZZ","FKWL","FL","FLAC","FLACU","FLAG","FLC","FLEX","FLGC","FLGT","FLIC","FLL","FLME","FLNC","FLNG","FLNT","FLO","FLR","FLS","FLT","FLUX","FLWS","FLXS","FLYA","FLYW","FMAO","FMBH","FMC","FMIV","FMIVW","FMN","FMNB","FMS","FMTX","FMX","FMY","FN","FNA","FNB","FNCB","FNCH","FND","FNF","FNGR","FNHC","FNKO","FNLC","FNV","FNVTW","FNWB","FNWD","FOA","FOCS","FOF","FOLD","FONR","FOR","FORA","FORD","FORG","FORM","FORR","FORTY","FOSL","FOSLL","FOUN","FOUNU","FOUNW","FOUR","FOX","FOXA","FOXF","FOXW","FPAC","FPAY","FPF","FPH","FPI","FPL","FR","FRA","FRAF","FRBA","FRBK","FRBN","FRBNW","FRC","FRD","FREE","FREQ","FREY","FRG","FRGAP","FRGE","FRGI","FRGT","FRHC","FRLAW","FRLN","FRME","FRMEP","FRO","FROG","FRON","FRONU","FRPH","FRPT","FRSG","FRSGW","FRSH","FRST","FRSX","FRT","FRWAW","FRXB","FSBC","FSBW","FSD","FSEA","FSFG","FSI","FSK","FSLR","FSLY","FSM","FSNB","FSP","FSR","FSRD","FSRDW","FSRX","FSS","FSSI","FSSIW","FST","FSTR","FSTX","FSV","FT","FTAA","FTAI","FTAIN","FTAIO","FTAIP","FTCH","FTCI","FTCV","FTCVU","FTCVW","FTDR","FTEK","FTEV","FTF","FTFT","FTHM","FTHY","FTI","FTK","FTNT","FTPA","FTPAU","FTRP","FTS","FTV","FTVI","FUBO","FUL","FULC","FULT","FULTP","FUN","FUNC","FUND","FURY","FUSB","FUSN","FUTU","FUV","FVAM","FVCB","FVIV","FVRR","FWBI","FWONA","FWONK","FWP","FWRD","FWRG","FXCO","FXCOR","FXLV","FXNC","FYBR","G","GAB","GABC","GACQ","GACQW","GAIA","GAIN","GAINN","GALT","GAM","GAMB","GAMC","GAME","GAN","GANX","GAPA","GAQ","GASS","GATEW","GATO","GATX","GAU","GB","GBAB","GBBK","GBBKR","GBBKW","GBCI","GBDC","GBIO","GBL","GBLI","GBNH","GBOX","GBR","GBRGR","GBS","GBT","GBTG","GBX","GCBC","GCI","GCMG","GCMGW","GCO","GCP","GCV","GD","GDDY","GDEN","GDL","GDNRW","GDO","GDOT","GDRX","GDS","GDV","GDYN","GE","GECC","GECCM","GECCN","GECCO","GEEX","GEEXU","GEF","GEG","GEGGL","GEHI","GEL","GENC","GENE","GENI","GEO","GEOS","GER","GERN","GES","GET","GEVO","GF","GFAI","GFAIW","GFF","GFGD","GFI","GFL","GFLU","GFS","GFX","GGAA","GGAAU","GGAAW","GGAL","GGB","GGE","GGG","GGMC","GGN","GGR","GGROW","GGT","GGZ","GH","GHAC","GHACU","GHC","GHG","GHIX","GHL","GHLD","GHM","GHRS","GHSI","GHY","GIB","GIC","GIFI","GIGM","GIII","GIIX","GIIXW","GIL","GILD","GILT","GIM","GIPR","GIPRW","GIS","GIW","GIWWW","GKOS","GL","GLAD","GLBE","GLBL","GLBS","GLDD","GLDG","GLEE","GLG","GLHA","GLLIR","GLLIW","GLMD","GLNG","GLO","GLOB","GLOP","GLP","GLPG","GLPI","GLQ","GLRE","GLS","GLSI","GLSPT","GLT","GLTO","GLU","GLUE","GLV","GLW","GLYC","GM","GMAB","GMBL","GMBLP","GMDA","GME","GMED","GMFI","GMGI","GMRE","GMS","GMTX","GMVD","GNAC","GNACU","GNE","GNFT","GNK","GNL","GNLN","GNPX","GNRC","GNS","GNSS","GNT","GNTX","GNTY","GNUS","GNW","GO","GOAC","GOBI","GOCO","GOED","GOEV","GOEVW","GOF","GOGL","GOGO","GOL","GOLD","GOLF","GOOD","GOODN","GOODO","GOOG","GOOGL","GOOS","GORO","GOSS","GOTU","GOVX","GP","GPAC","GPACU","GPACW","GPC","GPCO","GPCOW","GPI","GPJA","GPK","GPL","GPMT","GPN","GPOR","GPP","GPRE","GPRK","GPRO","GPS","GRAB","GRABW","GRAY","GRBK","GRC","GRCL","GRCYU","GREE","GREEL","GRF","GRFS","GRIL","GRIN","GRMN","GRNA","GRNAW","GRNQ","GROM","GROMW","GROV","GROW","GROY","GRPH","GRPN","GRTS","GRTX","GRVI","GRVY","GRWG","GRX","GS","GSAQ","GSAQW","GSAT","GSBC","GSBD","GSEV","GSHD","GSIT","GSK","GSL","GSLD","GSM","GSMG","GSQB","GSRM","GSRMU","GSUN","GSV","GT","GTAC","GTACU","GTBP","GTE","GTEC","GTES","GTH","GTHX","GTIM","GTLB","GTLS","GTN","GTPB","GTX","GTXAP","GTY","GUG","GURE","GUT","GVA","GVCIU","GVP","GWH","GWRE","GWRS","GWW","GXII","GXO","H","HA","HAAC","HAACU","HAACW","HAE","HAFC","HAIA","HAIAU","HAIAW","HAIN","HAL","HALL","HALO","HAPP","HARP","HAS","HASI","HAYN","HAYW","HBAN","HBANM","HBANP","HBB","HBCP","HBI","HBIO","HBM","HBNC","HBT","HCA","HCAR","HCARU","HCARW","HCAT","HCC","HCCI","HCDI","HCDIP","HCDIW","HCDIZ","HCI","HCIC","HCICU","HCII","HCKT","HCM","HCNE","HCNEU","HCNEW","HCP","HCSG","HCTI","HCVI","HCWB","HD","HDB","HDSN","HE","HEAR","HEES","HEI","HELE","HEP","HEPA","HEPS","HEQ","HERA","HERAU","HERAW","HES","HESM","HEXO","HFBL","HFFG","HFRO","HFWA","HGBL","HGEN","HGLB","HGTY","HGV","HHC","HHGCW","HHLA","HHS","HI","HIBB","HIE","HIG","HIGA","HIHO","HII","HIII","HIL","HILS","HIMS","HIMX","HIO","HIPO","HITI","HIVE","HIW","HIX","HL","HLBZ","HLBZW","HLF","HLG","HLGN","HLI","HLIO","HLIT","HLLY","HLMN","HLNE","HLT","HLTH","HLVX","HLX","HMC","HMCO","HMCOU","HMLP","HMN","HMNF","HMPT","HMST","HMTV","HMY","HNGR","HNI","HNNA","HNNAZ","HNRA","HNRG","HNST","HNVR","HNW","HOFT","HOFV","HOFVW","HOG","HOLI","HOLX","HOMB","HON","HONE","HOOD","HOOK","HOPE","HOTH","HOUR","HOUS","HOV","HOVNP","HOWL","HP","HPE","HPF","HPI","HPK","HPKEW","HPP","HPQ","HPS","HPX","HQH","HQI","HQL","HQY","HR","HRB","HRI","HRL","HRMY","HROW","HROWL","HRT","HRTG","HRTX","HRZN","HSAQ","HSBC","HSC","HSCS","HSDT","HSIC","HSII","HSKA","HSON","HST","HSTM","HSTO","HSY","HT","HTA","HTAQ","HTBI","HTBK","HTCR","HTD","HTFB","HTFC","HTGC","HTGM","HTH","HTHT","HTIA","HTIBP","HTLD","HTLF","HTLFP","HTOO","HTPA","HTY","HTZ","HTZWW","HUBB","HUBG","HUBS","HUDI","HUGE","HUGS","HUIZ","HUM","HUMA","HUMAW","HUN","HURC","HURN","HUSA","HUT","HUYA","HVBC","HVT","HWBK","HWC","HWCPZ","HWKN","HWKZ","HWM","HXL","HY","HYB","HYFM","HYI","HYLN","HYMC","HYMCW","HYMCZ","HYPR","HYRE","HYT","HYW","HYZN","HYZNW","HZN","HZNP","HZO","HZON","IAA","IAC","IACC","IAE","IAF","IAG","IART","IAS","IAUX","IBA","IBCP","IBER","IBEX","IBIO","IBKR","IBM","IBN","IBOC","IBP","IBRX","IBTX","ICAD","ICCC","ICCH","ICCM","ICD","ICE","ICFI","ICHR","ICL","ICLK","ICLR","ICMB","ICNC","ICPT","ICUI","ICVX","ID","IDA","IDAI","IDBA","IDCC","IDE","IDEX","IDN","IDR","IDRA","IDT","IDW","IDXX","IDYA","IE","IEA","IEAWW","IEP","IESC","IEX","IFBD","IFF","IFN","IFRX","IFS","IGA","IGAC","IGACW","IGC","IGD","IGI","IGIC","IGICW","IGMS","IGR","IGT","IGTAR","IH","IHD","IHG","IHIT","IHRT","IHS","IHT","IHTA","IIF","III","IIII","IIIIU","IIIIW","IIIN","IIIV","IIM","IINN","IINNW","IIPR","IIVI","IIVIP","IKNA","IKT","ILMN","ILPT","IMAB","IMAC","IMAQ","IMAQR","IMAQW","IMAX","IMBI","IMBIL","IMCC","IMCR","IMGN","IMGO","IMH","IMKTA","IMMP","IMMR","IMMX","IMNM","IMO","IMOS","IMPL","IMPP","IMPPP","IMPX","IMRA","IMRN","IMRX","IMTE","IMTX","IMUX","IMV","IMVT","IMXI","INAQ","INBK","INBKZ","INBX","INCR","INCY","INDB","INDI","INDIW","INDO","INDP","INDT","INFA","INFI","INFN","INFU","INFY","ING","INGN","INGR","INKA","INKAW","INKT","INM","INMB","INMD","INN","INNV","INO","INOD","INPX","INSE","INSG","INSI","INSM","INSP","INST","INSW","INT","INTA","INTC","INTEW","INTG","INTR","INTT","INTU","INTZ","INUV","INVA","INVE","INVH","INVO","INVZ","INVZW","INZY","IOBT","IONM","IONQ","IONR","IONS","IOSP","IOT","IOVA","IP","IPA","IPAR","IPAXW","IPDN","IPG","IPGP","IPHA","IPI","IPOD","IPOF","IPSC","IPVA","IPVF","IPVI","IPW","IPWR","IPX","IQ","IQI","IQMD","IQMDW","IQV","IR","IRBT","IRDM","IREN","IRIX","IRL","IRM","IRMD","IRNT","IRRX","IRS","IRT","IRTC","IRWD","IS","ISAA","ISD","ISDR","ISEE","ISIG","ISLE","ISLEW","ISO","ISPC","ISPO","ISPOW","ISR","ISRG","ISSC","ISTR","ISUN","IT","ITCB","ITCI","ITGR","ITHX","ITHXU","ITHXW","ITI","ITIC","ITOS","ITP","ITQ","ITRG","ITRI","ITRM","ITRN","ITT","ITUB","ITW","IVA","IVAC","IVC","IVCAU","IVCAW","IVCB","IVCBW","IVCP","IVDA","IVH","IVR","IVT","IVZ","IX","IXHL","IZEA","J","JACK","JAGX","JAKK","JAMF","JAN","JANX","JAQCW","JAZZ","JBGS","JBHT","JBI","JBL","JBLU","JBSS","JBT","JCE","JCI","JCIC","JCICW","JCSE","JCTCF","JD","JEF","JELD","JEMD","JEQ","JFIN","JFR","JFU","JG","JGGCU","JGGCW","JGH","JHAA","JHG","JHI","JHS","JHX","JILL","JJSF","JKHY","JKS","JLL","JLS","JMACW","JMIA","JMM","JMSB","JNCE","JNJ","JNPR","JOAN","JOB","JOBY","JOE","JOF","JOFF","JOFFU","JOFFW","JOUT","JPC","JPI","JPM","JPS","JPT","JQC","JRI","JRO","JRS","JRSH","JRVR","JSD","JSM","JSPR","JSPRW","JT","JUGG","JUGGW","JUN","JUPW","JUPWW","JVA","JWAC","JWEL","JWN","JWSM","JXN","JYAC","JYNT","JZXN","K","KACL","KACLR","KAHC","KAI","KAII","KAIR","KAL","KALA","KALU","KALV","KALWW","KAMN","KAR","KARO","KAVL","KB","KBAL","KBH","KBNT","KBNTW","KBR","KC","KCGI","KD","KDNY","KDP","KE","KELYA","KEN","KEP","KEQU","KERN","KERNW","KEX","KEY","KEYS","KF","KFFB","KFRC","KFS","KFY","KGC","KHC","KIDS","KIIIW","KIM","KIND","KINS","KINZ","KINZU","KINZW","KIO","KIQ","KIRK","KKR","KKRS","KLAC","KLAQ","KLAQU","KLIC","KLR","KLTR","KLXE","KMB","KMDA","KMF","KMI","KMPB","KMPH","KMPR","KMT","KMX","KN","KNBE","KNDI","KNOP","KNSA","KNSL","KNTE","KNTK","KNX","KO","KOD","KODK","KOF","KOP","KOPN","KORE","KOS","KOSS","KPLT","KPLTW","KPRX","KPTI","KR","KRBP","KRC","KREF","KRG","KRKR","KRMD","KRNL","KRNLU","KRNT","KRNY","KRO","KRON","KROS","KRP","KRT","KRTX","KRUS","KRYS","KSCP","KSM","KSPN","KSS","KT","KTB","KTCC","KTF","KTH","KTN","KTOS","KTRA","KTTA","KUKE","KULR","KURA","KVHI","KVSC","KW","KWAC","KWR","KXIN","KYCH","KYMR","KYN","KZIA","KZR","L","LAAA","LAB","LABP","LAC","LAD","LADR","LAKE","LAMR","LANC","LAND","LANDM","LANDO","LARK","LASR","LAUR","LAW","LAZ","LAZR","LAZY","LBAI","LBC","LBPH","LBRDA","LBRDK","LBRDP","LBRT","LBTYA","LBTYK","LC","LCA","LCAA","LCFY","LCFYW","LCI","LCID","LCII","LCNB","LCTX","LCUT","LCW","LDHA","LDHAW","LDI","LDOS","LDP","LE","LEA","LEAP","LECO","LEDS","LEE","LEG","LEGA","LEGH","LEGN","LEJU","LEN","LEO","LESL","LEU","LEV","LEVI","LEXX","LFAC","LFACU","LFACW","LFC","LFG","LFLY","LFLYW","LFMD","LFMDP","LFST","LFT","LFTR","LFUS","LFVN","LGAC","LGHL","LGHLW","LGI","LGIH","LGL","LGMK","LGND","LGO","LGST","LGSTW","LGTO","LGTOW","LGV","LGVN","LH","LHC","LHCG","LHDX","LHX","LI","LIAN","LIBYW","LICY","LIDR","LIDRW","LIFE","LII","LILA","LILAK","LILM","LILMW","LIN","LINC","LIND","LINK","LION","LIONW","LIQT","LITB","LITE","LITM","LITT","LIVE","LIVN","LIXT","LIZI","LJAQ","LJAQU","LJPC","LKCO","LKFN","LKQ","LL","LLAP","LLL","LLY","LMACA","LMACU","LMACW","LMAO","LMAT","LMB","LMDX","LMFA","LMND","LMNL","LMNR","LMPX","LMST","LMT","LNC","LND","LNDC","LNFA","LNG","LNN","LNSR","LNT","LNTH","LNW","LOAN","LOB","LOCL","LOCO","LODE","LOGC","LOGI","LOMA","LOOP","LOPE","LOTZ","LOTZW","LOV","LOVE","LOW","LPCN","LPG","LPI","LPL","LPLA","LPRO","LPSN","LPTH","LPTX","LPX","LQDA","LQDT","LRCX","LRFC","LRMR","LRN","LSAK","LSCC","LSEA","LSEAW","LSF","LSI","LSPD","LSTR","LSXMA","LSXMB","LSXMK","LTBR","LTC","LTCH","LTCHW","LTH","LTHM","LTRN","LTRPA","LTRX","LTRY","LTRYW","LU","LUCD","LULU","LUMN","LUMO","LUNA","LUNG","LUV","LUXA","LUXAU","LUXAW","LVAC","LVACW","LVLU","LVO","LVOX","LVRA","LVS","LVTX","LW","LWLG","LX","LXEH","LXFR","LXP","LXRX","LXU","LYB","LYEL","LYFT","LYG","LYL","LYLT","LYRA","LYT","LYTS","LYV","LZ","LZB","M","MA","MAA","MAAQ","MAAQW","MAC","MACA","MACAU","MACAW","MACC","MACK","MAG","MAIN","MAN","MANH","MANT","MANU","MAPS","MAPSW","MAQC","MAQCU","MAQCW","MAR","MARA","MARK","MARPS","MAS","MASI","MASS","MAT","MATV","MATW","MATX","MAV","MAX","MAXN","MAXR","MBAC","MBCN","MBI","MBII","MBIN","MBINN","MBINO","MBINP","MBIO","MBNKP","MBOT","MBRX","MBTCR","MBTCU","MBUU","MBWM","MC","MCAA","MCAAW","MCAC","MCB","MCBC","MCBS","MCD","MCFT","MCG","MCHP","MCHX","MCI","MCK","MCLD","MCN","MCO","MCR","MCRB","MCRI","MCS","MCW","MCY","MD","MDB","MDC","MDGL","MDGS","MDGSW","MDIA","MDJH","MDLZ","MDNA","MDRR","MDRX","MDT","MDU","MDV","MDVL","MDWD","MDWT","MDXG","MDXH","ME","MEAC","MEACW","MEC","MED","MEDP","MEDS","MEG","MEGI","MEI","MEIP","MEKA","MELI","MEOA","MEOAW","MEOH","MERC","MESA","MESO","MET","META","METC","METCL","METX","METXW","MF","MFA","MFC","MFD","MFG","MFGP","MFH","MFIN","MFM","MFV","MG","MGA","MGEE","MGF","MGI","MGIC","MGLD","MGM","MGNI","MGNX","MGPI","MGR","MGRB","MGRC","MGRD","MGTA","MGTX","MGU","MGY","MHD","MHF","MHH","MHI","MHK","MHLA","MHLD","MHN","MHNC","MHO","MHUA","MIC","MICS","MICT","MIDD","MIGI","MILE","MIMO","MIN","MIND","MINDP","MINM","MIO","MIR","MIRM","MIRO","MIST","MIT","MITC","MITK","MITO","MITQ","MITT","MIXT","MIY","MKC","MKD","MKFG","MKL","MKSI","MKTW","MKTX","ML","MLAB","MLAC","MLCO","MLI","MLKN","MLM","MLNK","MLP","MLR","MLSS","MLTX","MLVF","MMAT","MMC","MMD","MMI","MMLP","MMM","MMMB","MMP","MMS","MMSI","MMT","MMU","MMX","MMYT","MN","MNDO","MNDT","MNDY","MNKD","MNMD","MNOV","MNP","MNPR","MNRL","MNRO","MNSB","MNSBP","MNSO","MNST","MNTK","MNTS","MNTSW","MNTV","MNTX","MO","MOBQ","MOBQW","MOD","MODD","MODN","MODV","MOFG","MOGO","MOGU","MOH","MOHO","MOLN","MOMO","MON","MONCW","MOR","MORF","MORN","MOS","MOTS","MOV","MOVE","MOXC","MP","MPA","MPAA","MPACR","MPB","MPC","MPLN","MPLX","MPV","MPW","MPWR","MPX","MQ","MQT","MQY","MRAI","MRAM","MRBK","MRC","MRCC","MRCY","MREO","MRIN","MRK","MRKR","MRM","MRNA","MRNS","MRO","MRSN","MRTN","MRTX","MRUS","MRVI","MRVL","MS","MSA","MSAC","MSB","MSBI","MSC","MSCI","MSD","MSDA","MSDAW","MSEX","MSFT","MSGE","MSGM","MSGS","MSI","MSM","MSN","MSPR","MSPRW","MSPRZ","MSTR","MT","MTA","MTAC","MTACW","MTAL","MTB","MTBC","MTBCO","MTBCP","MTC","MTCH","MTCN","MTCR","MTD","MTDR","MTEK","MTEKW","MTEM","MTEX","MTG","MTH","MTLS","MTMT","MTN","MTNB","MTOR","MTP","MTR","MTRN","MTRX","MTRY","MTSI","MTTR","MTVC","MTW","MTX","MTZ","MU","MUA","MUC","MUDS","MUDSW","MUE","MUFG","MUI","MUJ","MULN","MUR","MURFW","MUSA","MUX","MVBF","MVF","MVIS","MVO","MVST","MVSTW","MVT","MWA","MX","MXC","MXCT","MXE","MXF","MXL","MYD","MYE","MYFW","MYGN","MYI","MYMD","MYN","MYNA","MYNZ","MYO","MYOV","MYPS","MYRG","MYSZ","MYTE","NAAC","NAACW","NAAS","NABL","NAC","NAD","NAII","NAK","NAN","NAOV","NAPA","NARI","NAT","NATH","NATI","NATR","NAUT","NAVB","NAVI","NAZ","NBB","NBEV","NBH","NBHC","NBIX","NBN","NBO","NBR","NBRV","NBSE","NBSTW","NBTB","NBTX","NBW","NBXG","NBY","NC","NCA","NCAC","NCACU","NCACW","NCLH","NCMI","NCNA","NCNO","NCR","NCSM","NCTY","NCV","NCZ","NDAC","NDACU","NDACW","NDAQ","NDLS","NDMO","NDP","NDRA","NDSN","NE","NEA","NECB","NEE","NEGG","NEM","NEN","NEO","NEOG","NEON","NEOV","NEP","NEPH","NEPT","NERV","NESR","NESRW","NET","NETI","NEU","NEWP","NEWR","NEWT","NEWTL","NEX","NEXA","NEXI","NEXT","NFBK","NFE","NFG","NFGC","NFJ","NFLX","NFYS","NG","NGC","NGD","NGG","NGL","NGM","NGMS","NGS","NGVC","NGVT","NH","NHC","NHI","NHIC","NHICW","NHS","NHTC","NHWK","NI","NIC","NICE","NICK","NID","NIE","NILE","NIM","NINE","NIO","NIQ","NISN","NIU","NJR","NKE","NKG","NKLA","NKSH","NKTR","NKTX","NKX","NL","NLIT","NLITU","NLITW","NLOK","NLS","NLSN","NLSP","NLSPW","NLTX","NLY","NM","NMAI","NMCO","NMFC","NMG","NMI","NMIH","NML","NMM","NMMC","NMR","NMRD","NMRK","NMS","NMT","NMTC","NMTR","NMZ","NN","NNBR","NNDM","NNI","NNN","NNOX","NNVC","NNY","NOA","NOAC","NOACW","NOAH","NOC","NODK","NOG","NOK","NOM","NOMD","NOTV","NOV","NOVA","NOVN","NOVT","NOW","NPAB","NPCE","NPCT","NPFD","NPK","NPO","NPTN","NPV","NQP","NR","NRACU","NRACW","NRBO","NRC","NRDS","NRDY","NREF","NRG","NRGV","NRGX","NRIM","NRIX","NRK","NRO","NRP","NRSN","NRSNW","NRT","NRUC","NRXP","NRXPW","NRZ","NS","NSA","NSC","NSIT","NSL","NSP","NSPR","NSR","NSS","NSSC","NSTB","NSTG","NSTS","NSYS","NTAP","NTB","NTCO","NTCT","NTES","NTG","NTGR","NTIC","NTIP","NTLA","NTNX","NTR","NTRA","NTRB","NTRBW","NTRS","NTRSO","NTST","NTUS","NTWK","NTZ","NU","NUE","NUO","NURO","NUS","NUTX","NUV","NUVA","NUVB","NUVL","NUW","NUWE","NUZE","NVACR","NVAX","NVCN","NVCR","NVCT","NVDA","NVEC","NVEE","NVEI","NVFY","NVG","NVGS","NVIV","NVMI","NVNO","NVO","NVOS","NVR","NVRO","NVS","NVSA","NVSAU","NVSAW","NVST","NVT","NVTA","NVTS","NVVE","NVVEW","NVX","NWBI","NWE","NWFL","NWG","NWL","NWLI","NWN","NWPX","NWS","NWSA","NX","NXC","NXDT","NXE","NXGL","NXGLW","NXGN","NXJ","NXN","NXP","NXPI","NXPL","NXRT","NXST","NXTC","NXTP","NYC","NYCB","NYMT","NYMTL","NYMTM","NYMTN","NYMTZ","NYMX","NYT","NYXH","NZF","O","OB","OBCI","OBE","OBLG","OBNK","OBSV","OC","OCAX","OCC","OCCI","OCCIO","OCFC","OCFT","OCG","OCGN","OCN","OCSL","OCUL","OCUP","OCX","ODC","ODFL","ODP","ODV","OEC","OEG","OEPW","OEPWW","OESX","OFC","OFG","OFIX","OFLX","OFS","OG","OGE","OGEN","OGI","OGN","OGS","OHI","OHPA","OHPAU","OHPAW","OI","OIA","OII","OIIM","OIS","OKE","OKTA","OKYO","OLB","OLED","OLIT","OLK","OLLI","OLMA","OLN","OLO","OLP","OLPX","OM","OMAB","OMC","OMCL","OMEG","OMER","OMEX","OMF","OMGA","OMI","OMIC","OMQS","ON","ONB","ONBPO","ONBPP","ONCR","ONCS","ONCT","ONCY","ONDS","ONEM","ONEW","ONL","ONON","ONTF","ONTO","ONTX","ONVO","ONYX","ONYXW","OOMA","OP","OPA","OPAD","OPBK","OPCH","OPEN","OPFI","OPGN","OPI","OPINL","OPK","OPNT","OPP","OPRA","OPRT","OPRX","OPT","OPTN","OPTT","OPY","OR","ORA","ORAN","ORC","ORCC","ORCL","ORGN","ORGNW","ORGO","ORGS","ORI","ORIC","ORLA","ORLY","ORMP","ORN","ORRF","ORTX","OSBC","OSCR","OSG","OSH","OSIS","OSK","OSPN","OSS","OST","OSTK","OSTR","OSTRU","OSTRW","OSUR","OSW","OTECW","OTEX","OTIC","OTIS","OTLK","OTLY","OTMO","OTRK","OTRKP","OTTR","OUST","OUT","OVBC","OVID","OVLY","OVV","OWL","OWLT","OXAC","OXACW","OXBR","OXBRW","OXLC","OXLCL","OXLCM","OXLCN","OXLCO","OXLCP","OXLCZ","OXM","OXSQ","OXSQG","OXSQL","OXUS","OXUSW","OXY","OYST","OZ","OZK","OZKAP","PAA","PAAS","PAC","PACB","PACI","PACK","PACW","PACWP","PACX","PACXU","PACXW","PAG","PAGP","PAGS","PAHC","PAI","PALI","PALT","PAM","PANL","PANW","PAQC","PAQCU","PAQCW","PAR","PARA","PARAA","PARAP","PARR","PASG","PATH","PATI","PATK","PAVM","PAVMZ","PAX","PAXS","PAY","PAYA","PAYC","PAYO","PAYS","PAYX","PB","PBA","PBBK","PBF","PBFS","PBFX","PBH","PBHC","PBI","PBLA","PBPB","PBR","PBT","PBTS","PBYI","PCAR","PCB","PCCT","PCF","PCG","PCGU","PCH","PCK","PCM","PCN","PCOR","PCPC","PCQ","PCRX","PCSA","PCSB","PCT","PCTI","PCTTW","PCTY","PCVX","PCX","PCYG","PCYO","PD","PDCE","PDCO","PDD","PDEX","PDFS","PDI","PDLB","PDM","PDO","PDOT","PDS","PDSB","PDT","PEAK","PEAR","PEARW","PEB","PEBK","PEBO","PECO","PED","PEG","PEGA","PEGR","PEGRU","PEGY","PEI","PEN","PENN","PEO","PEP","PEPG","PEPL","PEPLW","PERI","PESI","PETQ","PETS","PETV","PETVW","PETZ","PEV","PFBC","PFC","PFD","PFDR","PFDRW","PFE","PFG","PFGC","PFH","PFHC","PFHD","PFIE","PFIN","PFIS","PFL","PFLT","PFMT","PFN","PFO","PFS","PFSI","PFSW","PFTA","PFTAU","PFX","PFXNL","PG","PGC","PGEN","PGNY","PGP","PGR","PGRE","PGRU","PGRW","PGRWU","PGTI","PGY","PGYWW","PGZ","PH","PHAR","PHAS","PHAT","PHCF","PHD","PHG","PHGE","PHI","PHIC","PHIO","PHK","PHM","PHR","PHT","PHUN","PHUNW","PHVS","PHX","PI","PIAI","PICC","PII","PIII","PIIIW","PIK","PIM","PINC","PINE","PING","PINS","PIPP","PIPR","PIRS","PIXY","PJT","PK","PKBK","PKE","PKG","PKI","PKOH","PKX","PL","PLAB","PLAG","PLAY","PLBC","PLBY","PLCE","PLD","PLG","PLL","PLM","PLMI","PLMIU","PLMIW","PLMR","PLNT","PLOW","PLPC","PLRX","PLSE","PLTK","PLTR","PLUG","PLUS","PLX","PLXP","PLXS","PLYA","PLYM","PM","PMCB","PMD","PME","PMF","PMGM","PMGMW","PML","PMM","PMO","PMT","PMTS","PMVP","PMX","PNACU","PNBK","PNC","PNF","PNFP","PNFPP","PNI","PNM","PNNT","PNR","PNRG","PNT","PNTG","PNTM","PNW","POAI","PODD","POET","POLA","POLY","POND","PONO","PONOW","POOL","POR","PORT","POSH","POST","POW","POWI","POWL","POWRU","POWW","POWWP","PPBI","PPBT","PPC","PPG","PPIH","PPL","PPSI","PPT","PPTA","PRA","PRAA","PRAX","PRBM","PRCH","PRCT","PRDO","PRDS","PRE","PRFT","PRFX","PRG","PRGO","PRGS","PRI","PRIM","PRK","PRLB","PRLD","PRLH","PRM","PRMW","PRO","PROC","PROCW","PROF","PROV","PRPB","PRPC","PRPH","PRPL","PRPO","PRQR","PRS","PRSO","PRSR","PRSRU","PRSRW","PRT","PRTA","PRTG","PRTH","PRTK","PRTS","PRTY","PRU","PRVA","PRVB","PSA","PSAG","PSAGU","PSAGW","PSB","PSEC","PSF","PSFE","PSHG","PSMT","PSN","PSNL","PSNY","PSNYW","PSO","PSPC","PSTG","PSTH","PSTI","PSTL","PSTV","PSTX","PSX","PT","PTA","PTC","PTCT","PTE","PTEN","PTGX","PTIX","PTLO","PTMN","PTN","PTNR","PTON","PTPI","PTR","PTRA","PTRS","PTSI","PTVE","PTY","PUBM","PUCK","PUCKW","PUK","PULM","PUMP","PUYI","PV","PVBC","PVH","PVL","PW","PWFL","PWOD","PWP","PWR","PWSC","PWUPW","PX","PXD","PXLW","PXS","PXSAP","PYCR","PYN","PYPD","PYPL","PYR","PYS","PYT","PYXS","PZC","PZG","PZN","PZZA","QCOM","QCRH","QD","QDEL","QFIN","QFTA","QGEN","QH","QIPT","QK","QLGN","QLI","QLYS","QMCO","QNGY","QNRX","QNST","QQQX","QRHC","QRTEA","QRTEB","QRTEP","QRVO","QS","QSI","QSR","QTEK","QTEKW","QTNT","QTRX","QTT","QTWO","QUAD","QUBT","QUIK","QUMU","QUOT","QURE","QVCC","QVCD","R","RA","RAAS","RACE","RAD","RADA","RADI","RAIL","RAIN","RAM","RAMMU","RAMMW","RAMP","RANI","RAPT","RARE","RAVE","RBA","RBAC","RBB","RBBN","RBCAA","RBCN","RBKB","RBLX","RBOT","RC","RCA","RCAT","RCB","RCC","RCEL","RCFA","RCG","RCHG","RCHGU","RCHGW","RCI","RCII","RCKT","RCKY","RCL","RCLF","RCLFW","RCM","RCMT","RCON","RCOR","RCRT","RCRTW","RCS","RCUS","RDBX","RDBXW","RDCM","RDFN","RDHL","RDI","RDIB","RDN","RDNT","RDUS","RDVT","RDW","RDWR","RDY","RE","REAL","REAX","REE","REEAW","REED","REFI","REFR","REG","REGN","REI","REKR","RELI","RELIW","RELL","RELX","RELY","RENEU","RENEW","RENN","RENT","REPL","REPX","RERE","RES","RETA","RETO","REV","REVB","REVBW","REVEW","REVG","REVH","REVHU","REVHW","REX","REXR","REYN","REZI","RF","RFACW","RFI","RFIL","RFL","RFM","RFMZ","RFP","RGA","RGC","RGCO","RGEN","RGF","RGLD","RGLS","RGNX","RGP","RGR","RGS","RGT","RGTI","RGTIW","RH","RHE","RHI","RHP","RIBT","RICK","RIDE","RIG","RIGL","RILY","RILYG","RILYK","RILYL","RILYM","RILYN","RILYO","RILYP","RILYT","RILYZ","RIO","RIOT","RIV","RIVN","RJF","RKDA","RKLB","RKLY","RKT","RKTA","RL","RLAY","RLGT","RLI","RLJ","RLMD","RLTY","RLX","RLYB","RM","RMAX","RMBI","RMBL","RMBS","RMCF","RMD","RMED","RMGC","RMGCW","RMI","RMM","RMMZ","RMNI","RMO","RMR","RMT","RMTI","RNA","RNAZ","RNDB","RNER","RNERW","RNG","RNGR","RNLX","RNP","RNR","RNST","RNW","RNWK","RNWWW","RNXT","ROAD","ROC","ROCC","ROCGU","ROCK","ROCLU","ROCLW","ROG","ROIC","ROIV","ROIVW","ROK","ROKU","ROL","ROLL","ROLLP","RONI","ROOT","ROP","ROSE","ROSEU","ROSEW","ROSS","ROST","ROVR","RPAY","RPD","RPHM","RPID","RPM","RPRX","RPT","RPTX","RQI","RRBI","RRC","RRGB","RRR","RRX","RS","RSF","RSG","RSI","RSKD","RSLS","RSSS","RSVR","RSVRW","RTL","RTLPO","RTLPP","RTLR","RTX","RUBY","RUN","RUSHA","RUSHB","RUTH","RVAC","RVACU","RVACW","RVLP","RVLV","RVMD","RVNC","RVP","RVPH","RVPHW","RVSB","RVSN","RVT","RWAY","RWLK","RWT","RXDX","RXRX","RXST","RXT","RY","RYAAY","RYAM","RYAN","RYI","RYN","RYTM","RZA","RZB","RZLT","S","SA","SABR","SABRP","SABS","SABSW","SACC","SACH","SAFE","SAFM","SAFT","SAGA","SAGE","SAH","SAI","SAIA","SAIC","SAIL","SAITW","SAL","SALM","SAM","SAMAW","SAMG","SAN","SANA","SAND","SANM","SANW","SAP","SAR","SASI","SASR","SAT","SATL","SATLW","SATS","SAVA","SAVE","SB","SBAC","SBBA","SBCF","SBET","SBEV","SBFG","SBFM","SBFMW","SBGI","SBH","SBI","SBIG","SBII","SBLK","SBNY","SBNYP","SBOW","SBR","SBRA","SBS","SBSI","SBSW","SBT","SBTX","SBUX","SCAQU","SCCB","SCCC","SCCE","SCCF","SCCO","SCD","SCHL","SCHN","SCHW","SCI","SCKT","SCL","SCLE","SCLEU","SCLEW","SCM","SCOA","SCOAW","SCOB","SCOBU","SCOBW","SCOR","SCPH","SCPL","SCPS","SCRM","SCRMW","SCS","SCSC","SCTL","SCU","SCVL","SCWO","SCWX","SCX","SCYX","SD","SDAC","SDACU","SDACW","SDC","SDGR","SDH","SDHY","SDIG","SDPI","SE","SEAC","SEAS","SEAT","SEB","SECO","SEDG","SEE","SEED","SEEL","SEER","SEIC","SELB","SELF","SEM","SEMR","SENEA","SENS","SERA","SES","SESN","SEV","SEVN","SF","SFB","SFBS","SFE","SFET","SFIX","SFL","SFM","SFNC","SFST","SFT","SG","SGA","SGBX","SGC","SGEN","SGFY","SGH","SGHC","SGHL","SGHT","SGIIW","SGLY","SGMA","SGML","SGMO","SGRP","SGRY","SGTX","SGU","SHAC","SHAK","SHBI","SHC","SHCA","SHCAU","SHCR","SHCRW","SHEL","SHEN","SHG","SHI","SHIP","SHLS","SHLX","SHO","SHOO","SHOP","SHPW","SHQA","SHQAU","SHW","SHYF","SI","SIBN","SID","SIDU","SIEB","SIEN","SIER","SIF","SIFY","SIG","SIGA","SIGI","SIGIP","SII","SILC","SILK","SILV","SIMO","SINT","SIOX","SIRE","SIRI","SISI","SITC","SITE","SITM","SIVB","SIVBP","SIX","SJ","SJI","SJIJ","SJIV","SJM","SJR","SJT","SJW","SKE","SKIL","SKIN","SKLZ","SKM","SKT","SKX","SKY","SKYA","SKYH","SKYT","SKYW","SKYX","SLAB","SLAC","SLB","SLCA","SLCR","SLCRW","SLDB","SLDP","SLDPW","SLF","SLG","SLGC","SLGG","SLGL","SLGN","SLHG","SLHGP","SLI","SLM","SLN","SLNH","SLNHP","SLNO","SLP","SLQT","SLRC","SLRX","SLS","SLVM","SLVR","SLVRU","SM","SMAP","SMAPW","SMAR","SMBC","SMBK","SMCI","SMED","SMFG","SMFL","SMFR","SMFRW","SMG","SMHI","SMID","SMIHU","SMIT","SMLP","SMLR","SMM","SMMF","SMMT","SMP","SMPL","SMR","SMRT","SMSI","SMTC","SMTI","SMTS","SMWB","SNA","SNAP","SNAX","SNAXW","SNBR","SNCE","SNCR","SNCRL","SNCY","SND","SNDA","SNDL","SNDR","SNDX","SNES","SNEX","SNFCA","SNGX","SNMP","SNN","SNOA","SNOW","SNP","SNPO","SNPS","SNPX","SNRH","SNRHU","SNRHW","SNSE","SNT","SNTG","SNTI","SNV","SNX","SNY","SO","SOBR","SOFI","SOFO","SOHO","SOHOB","SOHON","SOHOO","SOHU","SOI","SOJC","SOJD","SOJE","SOL","SOLN","SOLO","SON","SOND","SONM","SONN","SONO","SONX","SONY","SOPA","SOPH","SOR","SOS","SOTK","SOUN","SOUNW","SOVO","SP","SPB","SPCB","SPCE","SPE","SPFI","SPG","SPGI","SPGS","SPH","SPI","SPIR","SPKB","SPKBU","SPKBW","SPLK","SPLP","SPNE","SPNS","SPNT","SPOK","SPOT","SPPI","SPR","SPRB","SPRC","SPRO","SPSC","SPT","SPTK","SPTKW","SPTN","SPWH","SPWR","SPXC","SPXX","SQ","SQFT","SQFTP","SQFTW","SQL","SQLLW","SQM","SQNS","SQSP","SQZ","SR","SRAD","SRAX","SRC","SRCE","SRCL","SRDX","SRE","SREA","SREV","SRG","SRGA","SRI","SRL","SRLP","SRNE","SRPT","SRRK","SRSA","SRT","SRTS","SRV","SRZN","SRZNW","SSAA","SSB","SSBI","SSBK","SSD","SSIC","SSKN","SSL","SSNC","SSNT","SSP","SSRM","SSSS","SST","SSTI","SSTK","SSU","SSY","SSYS","ST","STAA","STAB","STAF","STAG","STAR","STBA","STC","STCN","STE","STEM","STEP","STER","STEW","STG","STGW","STIM","STK","STKL","STKS","STLA","STLD","STM","STN","STNE","STNG","STOK","STON","STOR","STR","STRA","STRC","STRCW","STRE","STRL","STRM","STRN","STRNW","STRO","STRR","STRS","STRT","STRY","STSA","STSS","STSSW","STT","STTK","STVN","STWD","STX","STXS","STZ","SU","SUAC","SUI","SUM","SUMO","SUN","SUNL","SUNW","SUP","SUPN","SUPV","SURF","SURG","SURGW","SUZ","SVC","SVFA","SVFAU","SVFAW","SVFD","SVM","SVNAW","SVRA","SVRE","SVT","SVVC","SWAG","SWAV","SWBI","SWCH","SWET","SWETW","SWI","SWIM","SWIR","SWK","SWKH","SWKS","SWN","SWT","SWTX","SWVL","SWVLW","SWX","SWZ","SXC","SXI","SXT","SXTC","SY","SYBT","SYBX","SYF","SYK","SYM","SYN","SYNA","SYNH","SYNL","SYPR","SYRS","SYTA","SYTAW","SYY","SZC","T","TA","TAC","TACT","TAIT","TAK","TAL","TALK","TALKW","TALO","TALS","TANH","TANNI","TANNL","TANNZ","TAOP","TAP","TARA","TARO","TARS","TASK","TAST","TATT","TAYD","TBB","TBBK","TBC","TBCPU","TBI","TBK","TBKCP","TBLA","TBLD","TBLT","TBLTW","TBNK","TBPH","TC","TCBC","TCBI","TCBIO","TCBK","TCBP","TCBPW","TCBS","TCBX","TCDA","TCFC","TCI","TCMD","TCN","TCOM","TCON","TCPC","TCRR","TCRT","TCRX","TCS","TCVA","TCX","TD","TDC","TDCX","TDF","TDG","TDOC","TDS","TDUP","TDW","TDY","TEAF","TEAM","TECH","TECK","TECTP","TEDU","TEF","TEI","TEKK","TEKKU","TEL","TELA","TELL","TELZ","TEN","TENB","TENX","TEO","TER","TERN","TESS","TETC","TETCU","TETCW","TETE","TETEU","TEVA","TEX","TFC","TFFP","TFII","TFSA","TFSL","TFX","TG","TGA","TGAA","TGAN","TGB","TGH","TGI","TGLS","TGNA","TGR","TGS","TGT","TGTX","TGVC","TGVCW","TH","THACW","THC","THCA","THCP","THFF","THG","THM","THMO","THO","THQ","THR","THRM","THRN","THRX","THRY","THS","THTX","THW","THWWW","TIG","TIGO","TIGR","TIL","TILE","TIMB","TINV","TIPT","TIRX","TISI","TITN","TIVC","TIXT","TJX","TK","TKAT","TKC","TKLF","TKNO","TKR","TLGA","TLGY","TLGYW","TLIS","TLK","TLRY","TLS","TLSA","TLYS","TM","TMAC","TMBR","TMC","TMCI","TMCWW","TMDI","TMDX","TME","TMHC","TMKR","TMKRU","TMKRW","TMO","TMP","TMQ","TMST","TMUS","TMX","TNC","TNDM","TNET","TNGX","TNK","TNL","TNON","TNP","TNXP","TNYA","TOI","TOIIW","TOL","TOMZ","TOP","TOPS","TOST","TOUR","TOWN","TPB","TPC","TPG","TPGY","TPH","TPHS","TPIC","TPL","TPR","TPST","TPTA","TPTX","TPVG","TPX","TPZ","TR","TRAQ","TRC","TRCA","TRDA","TREE","TREX","TRGP","TRHC","TRI","TRIB","TRIN","TRIP","TRKA","TRMB","TRMD","TRMK","TRMR","TRN","TRNO","TRNS","TRON","TROO","TROW","TROX","TRP","TRQ","TRS","TRST","TRT","TRTL","TRTN","TRTX","TRU","TRUE","TRUP","TRV","TRVG","TRVI","TRVN","TRX","TS","TSAT","TSBK","TSCO","TSE","TSEM","TSHA","TSI","TSIB","TSLA","TSLX","TSM","TSN","TSP","TSPQ","TSQ","TSRI","TSVT","TT","TTC","TTCF","TTD","TTE","TTEC","TTEK","TTGT","TTI","TTM","TTMI","TTNP","TTOO","TTP","TTSH","TTWO","TU","TUEM","TUFN","TUP","TURN","TUSK","TUYA","TV","TVC","TVE","TVTX","TW","TWI","TWIN","TWKS","TWLO","TWLV","TWN","TWND","TWNI","TWNK","TWO","TWOA","TWOU","TWST","TWTR","TX","TXG","TXMD","TXN","TXRH","TXT","TY","TYDE","TYG","TYL","TYME","TYRA","TZOO","TZPS","TZPSW","U","UA","UAA","UAL","UAMY","UAN","UAVS","UBA","UBCP","UBER","UBFO","UBOH","UBP","UBS","UBSI","UBX","UCBI","UCBIO","UCL","UCTT","UDMY","UDR","UE","UEC","UEIC","UFAB","UFCS","UFI","UFPI","UFPT","UG","UGI","UGIC","UGP","UGRO","UHAL","UHS","UHT","UI","UIHC","UIS","UK","UKOMW","UL","ULBI","ULCC","ULH","ULTA","UMBF","UMC","UMH","UMPQ","UNAM","UNB","UNCY","UNF","UNFI","UNH","UNIT","UNM","UNMA","UNP","UNTY","UNVR","UONE","UONEK","UP","UPC","UPH","UPLD","UPS","UPST","UPTDW","UPWK","URBN","URG","URGN","URI","UROY","USA","USAC","USAK","USAP","USAS","USAU","USB","USCB","USCT","USDP","USEA","USEG","USER","USFD","USIO","USLM","USM","USNA","USPH","USWS","USWSW","USX","UTAA","UTAAW","UTF","UTG","UTHR","UTI","UTL","UTMD","UTME","UTRS","UTSI","UTZ","UUU","UUUU","UVE","UVSP","UVV","UWMC","UXIN","UZD","UZE","UZF","V","VABK","VAC","VACC","VAL","VALE","VALN","VALU","VAPO","VATE","VAXX","VBF","VBIV","VBLT","VBNK","VBTX","VC","VCEL","VCIF","VCKA","VCKAW","VCNX","VCSA","VCTR","VCV","VCXA","VCXAU","VCXAW","VCXB","VCYT","VECO","VECT","VEDU","VEEE","VEEV","VEL","VELO","VELOU","VENA","VENAR","VENAW","VEON","VERA","VERB","VERBW","VERI","VERO","VERU","VERV","VERX","VERY","VET","VEV","VFC","VFF","VFL","VG","VGFC","VGI","VGM","VGR","VGZ","VHAQ","VHC","VHI","VHNAW","VIA","VIAO","VIASP","VIAV","VICI","VICR","VIEW","VIEWW","VIGL","VINC","VINE","VINO","VINP","VIOT","VIPS","VIR","VIRC","VIRI","VIRT","VIRX","VISL","VIST","VITL","VIV","VIVE","VIVK","VIVO","VJET","VKI","VKQ","VKTX","VLAT","VLCN","VLD","VLDR","VLDRW","VLGEA","VLN","VLNS","VLO","VLON","VLRS","VLT","VLTA","VLY","VLYPO","VLYPP","VMAR","VMC","VMCAW","VMD","VMEO","VMGA","VMI","VMO","VMW","VNCE","VNDA","VNET","VNO","VNOM","VNRX","VNT","VNTR","VOC","VOD","VOR","VORB","VORBW","VOXX","VOYA","VPG","VPV","VQS","VRA","VRAR","VRAY","VRCA","VRDN","VRE","VREX","VRM","VRME","VRMEW","VRNA","VRNS","VRNT","VRPX","VRRM","VRSK","VRSN","VRT","VRTS","VRTV","VRTX","VS","VSACW","VSAT","VSCO","VSEC","VSH","VST","VSTA","VSTM","VSTO","VTAQ","VTAQW","VTEX","VTGN","VTIQ","VTIQW","VTN","VTNR","VTOL","VTR","VTRS","VTRU","VTSI","VTVT","VTYX","VUZI","VVI","VVNT","VVOS","VVPR","VVR","VVV","VWE","VWEWW","VXRT","VYGG","VYGR","VYNE","VYNT","VZ","VZIO","VZLA","W","WAB","WABC","WAFD","WAFDP","WAFU","WAL","WALD","WALDW","WARR","WASH","WAT","WATT","WAVC","WAVD","WAVE","WB","WBA","WBD","WBEV","WBS","WBT","WBX","WCC","WCN","WD","WDAY","WDC","WDFC","WDH","WDI","WDS","WE","WEA","WEAV","WEBR","WEC","WEJO","WEJOW","WEL","WELL","WEN","WERN","WES","WETF","WEX","WEYS","WF","WFC","WFCF","WFG","WFRD","WGO","WH","WHD","WHF","WHG","WHLM","WHLR","WHLRD","WHLRP","WHR","WIA","WILC","WIMI","WINA","WING","WINT","WINVR","WIRE","WISA","WISH","WIT","WIW","WIX","WK","WKEY","WKHS","WKME","WKSP","WKSPW","WLDN","WLFC","WLK","WLKP","WLMS","WLY","WM","WMB","WMC","WMG","WMK","WMPN","WMS","WMT","WNC","WNEB","WNNR","WNS","WNW","WOLF","WOOF","WOR","WORX","WOW","WPC","WPCA","WPCB","WPM","WPP","WPRT","WQGA","WRAP","WRB","WRBY","WRE","WRK","WRLD","WRN","WSBC","WSBCP","WSBF","WSC","WSFS","WSM","WSO","WSR","WST","WSTG","WTBA","WTER","WTFC","WTFCM","WTFCP","WTI","WTM","WTRG","WTRH","WTS","WTT","WTTR","WTW","WU","WULF","WVE","WVVI","WVVIP","WW","WWAC","WWACW","WWD","WWE","WWR","WWW","WY","WYNN","WYY","X","XAIR","XBIO","XBIT","XCUR","XEL","XELA","XELAP","XELB","XENE","XERS","XFIN","XFINW","XFLT","XFOR","XGN","XHR","XIN","XL","XLO","XM","XMTR","XNCR","XNET","XOM","XOMA","XOMAO","XOMAP","XOS","XOSWW","XP","XPAX","XPAXW","XPDB","XPDBU","XPDBW","XPEL","XPER","XPEV","XPL","XPO","XPOA","XPOF","XPON","XPRO","XRAY","XRTX","XRX","XSPA","XTLB","XTNT","XXII","XYF","XYL","Y","YALA","YCBD","YELL","YELP","YETI","YEXT","YGMZ","YI","YJ","YMAB","YMM","YMTX","YORW","YOTAR","YOTAW","YOU","YPF","YQ","YRD","YSG","YTEN","YTPG","YTRA","YUM","YUMC","YVR","YY","Z","ZBH","ZBRA","ZCMD","ZD","ZDGE","ZEAL","ZEN","ZENV","ZEPP","ZEST","ZETA","ZEUS","ZEV","ZG","ZGN","ZH","ZI","ZIM","ZIMV","ZING","ZINGW","ZION","ZIONL","ZIONO","ZIONP","ZIP","ZIVO","ZKIN","ZLAB","ZM","ZNH","ZNTL","ZOM","ZS","ZT","ZTEK","ZTO","ZTR","ZTS","ZUMZ","ZUO","ZVIA","ZVO","ZWRK","ZWS","ZY","ZYME","ZYNE","ZYXI"
    ])
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
