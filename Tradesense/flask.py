import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random

# User data file
USER_DATA_FILE = "users.json"

# Email configuration (Replace with your email and password)
EMAIL_ADDRESS = "tradesense2003@gmail.com"  # Replace with your Gmail email
EMAIL_PASSWORD = "bows negp rtlt ngqs"  # Replace with your Gmail app-specific password

# Load/save users
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f)

# Send email function
def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)} 📧")
        return False

# Initialize session state with all necessary variables
def initialize_session_state():
    defaults = {
        'logged_in': False,
        'username': "",
        'email': "",
        'users': load_users(),
        'candle_data': pd.DataFrame(columns=["time", "open", "high", "low", "close"]),
        'trading_active': False,
        'symbol': "AAPL",
        'last_price': {},
        'sold_price': {},
        'bought_price': {},
        'last_update_time': 0,
        'stock_data_cache': {},
        'company_name_cache': {},
        'portfolio_history': [],
        'current_price': 0.0,
        'market_news': [],
        'market_movers': {"gainers": []},
        'recent_data': pd.DataFrame(),
        'price_alerts': [],
        'alert_popup': False,
        'alert_message': "",
        'buy_message': "",
        'sell_message': "",
        'popup_start_time': time.time(),
        'show_popup': True,
        'watchlist_last_update': 0,
        'show_register': False,
        'show_why_traderiser': False,
        'news_last_update': 0,
        'learning_last_update': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Preload essential data at startup with optimization
def preload_data():
    if st.session_state.recent_data.empty or not st.session_state.market_news or not st.session_state.market_movers["gainers"]:
        st.session_state.recent_data = fetch_recent_data()
        st.session_state.market_news = fetch_market_news()
        st.session_state.market_movers = fetch_market_movers()

# Fetch stock data from yfinance with caching
def get_stock_data(symbol, period="1d", interval="1m"):
    cache_key = f"{symbol}_{period}_{interval}"
    if cache_key in st.session_state.stock_data_cache:
        return st.session_state.stock_data_cache[cache_key]

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            df = pd.DataFrame([{
                "time": datetime.now(),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.2
            }])
        else:
            df = df.reset_index().rename(columns={"Datetime": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
        st.session_state.stock_data_cache[cache_key] = df
        return df
    except Exception:
        return pd.DataFrame([{
            "time": datetime.now(),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2
        }])

# Fetch company name with caching
def get_company_name(symbol):
    if symbol in st.session_state.company_name_cache:
        return st.session_state.company_name_cache[symbol]

    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        company_name = info.get("longName", "Unknown Company")
        st.session_state.company_name_cache[symbol] = company_name
        return company_name
    except Exception:
        st.session_state.company_name_cache[symbol] = "Unknown Company"
        return "Unknown Company"

# Fetch current price for a symbol
def get_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d", interval="1m")
        if not data.empty:
            return round(data["Close"][-1], 2)
        return 0.0
    except Exception:
        return 0.0

# Fetch market news using yfinance or random fallback
def fetch_market_news():
    news_options = [
        "Tech stocks rally as AI demand surges. 🚀",
        "Fed signals potential rate cuts in Q2 2025. 📉",
        "Crypto market sees 10% surge overnight. 💰",
        "Oil prices drop amid geopolitical tensions. 🛢️",
        "Retail sector booms with holiday sales up 15%. 🛍️",
        "Semiconductor shortage eases, stocks soar. 💾",
        "Green energy investments hit record highs. 🌍",
        "Global markets mixed after inflation data release. 📊",
        "Pharma stocks rise on new drug approvals. 💊",
        "Automakers pivot to EVs, boosting shares. 🚗"
    ]
    try:
        sp500 = yf.Ticker("^GSPC")
        news = sp500.news[:5]
        news_items = [item["title"] for item in news]
        if not news_items:
            return random.sample(news_options, 5)
        return news_items
    except Exception:
        return random.sample(news_options, 5)

# Fetch market movers (only Top Gainers) with optimization
def fetch_market_movers():
    symbols = ["AAPL", "TSLA", "NVDA", "META", "GOOGL", "MSFT", "AMZN", "AMD", "INTC", "PYPL"]
    gainers = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d")
            if not data.empty:
                change = ((data["Close"][-1] - data["Open"][0]) / data["Open"][0]) * 100
                change = round(change, 2)
                if change >= 0:
                    gainers.append({"symbol": symbol, "change": change})
        except Exception:
            continue
    gainers = sorted(gainers, key=lambda x: x["change"], reverse=True)[:5]
    if not gainers:
        gainers = [{"symbol": f"GAINER{i}", "change": 3.5 - i*0.2} for i in range(5)]
    return {"gainers": gainers}

# Fetch recent data for selected stocks with optimization
def fetch_recent_data():
    symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "AMD", "INTC", "PYPL"]
    data_list = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d", interval="1m")
            info = stock.info
            if not data.empty:
                latest = data.iloc[-1]
                change = ((latest["Close"] - data.iloc[0]["Open"]) / data.iloc[0]["Open"]) * 100
                data_list.append({
                    "Symbol": symbol,
                    "Company": info.get("longName", "Unknown Company"),
                    "Price": round(latest["Close"], 2),
                    "Volume": int(latest["Volume"]),
                    "Change %": round(change, 2),
                    "52w high": round(info.get("fiftyTwoWeekHigh", 0), 2),
                    "52w low": round(info.get("fiftyTwoWeekLow", 0), 2),
                    "Market cap (B)": round(info.get("marketCap", 0) / 1e9, 2),
                    "P/e ratio": round(info.get("trailingPE", 0), 2),
                    "Dividend yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0.0,
                    "Eps": round(info.get("trailingEps", 0), 2)
                })
        except Exception:
            data_list.append({
                "Symbol": symbol,
                "Company": "Unknown Company",
                "Price": 100.0,
                "Volume": 1000000,
                "Change %": 0.0,
                "52w high": 110.0,
                "52w low": 90.0,
                "Market cap (B)": 100.0,
                "P/e ratio": 15.0,
                "Dividend yield": 1.5,
                "Eps": 5.0
            })
    df = pd.DataFrame(data_list)
    df.index = range(1, len(df) + 1)
    return df

# Fetch watchlist data with real-time updates
def fetch_watchlist_data(symbols):
    data_list = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d", interval="1m")
            info = stock.info
            if not data.empty:
                latest = data.iloc[-1]
                data_list.append({
                    "Ticker symbol": symbol,
                    "Company name": info.get("longName", "Unknown Company"),
                    "Price": round(latest["Close"], 2),
                    "Volume": int(latest["Volume"]),
                    "Industry": info.get("industry", "N/A"),
                    "Market cap": round(info.get("marketCap", 0) / 1e9, 2),
                    "P/e ratio": round(info.get("trailingPE", 0), 2)
                })
        except Exception:
            data_list.append({
                "Ticker symbol": symbol,
                "Company name": "Unknown Company",
                "Price": 100.0,
                "Volume": 1000000,
                "Industry": "N/A",
                "Market cap": 100.0,
                "P/e ratio": 15.0
            })
    df = pd.DataFrame(data_list)
    df.index = range(1, len(df) + 1)
    return df

# Simulate real-time candle updates
def update_candle_data(symbol):
    data = get_stock_data(symbol, period="1d", interval="1m")
    if not data.empty:
        latest_candle = data.iloc[-1]
        volatility = (data["high"].max() - data["low"].min()) * 0.05 or 0.5
        new_open = st.session_state.last_price.get(symbol, latest_candle["close"])
    else:
        new_open = st.session_state.last_price.get(symbol, 100.0)
        volatility = 0.5

    new_high = new_open + np.random.uniform(0, volatility)
    new_low = new_open - np.random.uniform(0, volatility)
    new_close = new_open + np.random.uniform(-volatility * 0.3, volatility * 0.3)
    
    new_candle = {
        "time": datetime.now(),
        "open": new_open,
        "high": new_high,
        "low": new_low,
        "close": new_close
    }
    st.session_state.candle_data = pd.concat(
        [st.session_state.candle_data, pd.DataFrame([new_candle])], ignore_index=True
    )
    if len(st.session_state.candle_data) > 15:
        st.session_state.candle_data = st.session_state.candle_data.iloc[-15:]
    st.session_state.last_price[symbol] = new_close
    st.session_state.current_price = new_close

# Check price alerts and send email
def check_price_alerts():
    if not st.session_state.logged_in:
        return
    user_data = st.session_state.users.get(st.session_state.username, {})
    if not user_data:
        return
    email = user_data.get("email", "")
    if not email:
        return
    alerts_to_remove = []
    for alert in st.session_state.price_alerts:
        symbol = alert.get("symbol", "")
        target_price = alert.get("target_price", 0.0)
        current_price = st.session_state.last_price.get(symbol, get_current_price(symbol))
        if current_price >= target_price:
            alert_message = f"Price Alert! 📢 {symbol} has reached ${current_price:.2f} (Target: ${target_price:.2f}) 🎯"
            st.session_state.alert_message = alert_message
            st.session_state.alert_popup = True
            subject = f"TradeRiser Price Alert: {symbol} 🚀"
            body = f"""
🌟 Dear {st.session_state.username}, 🌟

🎉 Great news! Your price alert for {symbol} has been triggered! 🎉

📊 {symbol} has reached ${current_price:.2f} (Target: ${target_price:.2f}) 🚀

💡 Keep an eye on the market and make your next move! Trade smart with TradeRiser! 📈

Happy Trading! 😊
The TradeRiser Team 🌟
"""
            send_email(email, subject, body)
            alerts_to_remove.append(alert)
    for alert in alerts_to_remove:
        st.session_state.price_alerts.remove(alert)
    user_data["price_alerts"] = st.session_state.price_alerts
    save_users(st.session_state.users)

# Login Page
def login():
    st.markdown("""
        <style>
        body {
            background: linear-gradient(45deg, #0d1b2a, #1b263b, #2a1a3e, #1a2e2a);
            background-size: 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Roboto', sans-serif;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .ticker-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }
        .ticker {
            position: absolute;
            top: 20px;
            white-space: nowrap;
            font-size: 16px;
            color: #00ffcc;
            opacity: 0.7;
            animation: tickerMove 25s linear infinite;
        }
        @keyframes tickerMove {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        .particle {
            position: absolute;
            width: 5px;
            height: 5px;
            background: #00ffcc;
            border-radius: 50%;
            opacity: 0.4;
            animation: particleOrbit 8s infinite ease-in-out;
        }
        @keyframes particleOrbit {
            0% { transform: translateY(0) scale(1); opacity: 0.4; }
            25% { transform: translateX(20px) translateY(-50vh) scale(0.8); opacity: 0.6; }
            50% { transform: translateX(-20px) translateY(-100vh) scale(0.5); opacity: 0.2; }
            75% { transform: translateX(10px) translateY(-50vh) scale(0.8); opacity: 0.6; }
            100% { transform: translateY(0) scale(1); opacity: 0.4; }
        }
        @keyframes neonPulse {
            0% { text-shadow: 0 0 5px #00ffcc, 0 0 10px #00ffcc, 0 0 15px #00ffcc; }
            50% { text-shadow: 0 0 15px #00ffcc, 0 0 25px #00ffcc, 0 0 35px #00ffcc; }
            100% { text-shadow: 0 0 5px #00ffcc, 0 0 10px #00ffcc, 0 0 15px #00ffcc; }
        }
        @keyframes techyBounce {
            0% { opacity: 0; transform: translateY(30px) scale(0.95); }
            60% { opacity: 1; transform: translateY(-10px) scale(1.05); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes orbitGlow {
            0% { box-shadow: 0 0 5px #00ffcc, inset 0 0 5px #00ffcc; }
            50% { box-shadow: 0 0 20px #00ffcc, inset 0 0 10px #00ffcc; }
            100% { box-shadow: 0 0 5px #00ffcc, inset 0 0 5px #00ffcc; }
        }
        .login-title {
            color: #00ffcc;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
            margin-bottom: 15px;
            animation: neonPulse 1.5s infinite;
            text-align: center;
        }
        .welcome-text {
            color: #e0e0e0;
            font-size: 20px;
            margin: 20px auto;
            max-width: 700px;
            animation: techyBounce 1s ease-in-out;
            text-align: center;
            line-height: 1.5;
        }
        .features-section {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin: 40px auto;
            max-width: 1200px;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            width: 200px;
            text-align: center;
            animation: techyBounce 1.2s ease-in-out;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            border: 1px solid rgba(0, 255, 204, 0.2);
        }
        .feature-card:hover {
            transform: translateY(-10px) rotate(2deg);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.6), inset 0 0 10px rgba(0, 255, 255, 0.3);
            animation: orbitGlow 1s infinite;
        }
        .feature-icon {
            font-size: 32px;
            color: #00ffcc;
            margin-bottom: 10px;
            animation: spinPulse 3s infinite ease-in-out;
        }
        @keyframes spinPulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2) rotate(10deg); }
            100% { transform: scale(1); }
        }
        .feature-title {
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
        }
        .feature-desc {
            color: #b0b0b0;
            font-size: 14px;
        }
        .market-stats {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 40px auto;
            flex-wrap: wrap;
            max-width: 800px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 15px;
            width: 150px;
            text-align: center;
            animation: techyBounce 1.4s ease-in-out;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            border: 1px solid rgba(0, 255, 204, 0.2);
        }
        .stat-card:hover {
            transform: translateY(-10px) scale(1.05);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
            animation: orbitGlow 1s infinite;
        }
        .stat-title {
            color: #00ffcc;
            font-size: 14px;
        }
        .stat-value {
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
        }
        .testimonials-section {
            margin: 40px auto;
            padding: 0 20px;
            max-width: 800px;
        }
        .testimonial {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin: 20px auto;
            max-width: 600px;
            animation: techyBounce 1.6s ease-in-out;
            border-left: 4px solid #00ffcc;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .testimonial:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.5);
            animation: orbitGlow 1s infinite;
        }
        .testimonial-text {
            color: #e0e0e0;
            font-size: 16px;
            font-style: italic;
            margin-bottom: 10px;
        }
        .testimonial-author {
            color: #00ffcc;
            font-size: 14px;
            text-align: right;
            font-weight: bold;
        }
        .why-traderiser-section {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 25px;
            margin: 20px auto;
            max-width: 800px;
            text-align: center;
            animation: techyBounce 1s ease-in-out;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .why-traderiser-section:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
            animation: orbitGlow 1s infinite;
        }
        .why-traderiser-title {
            color: #00ffcc;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
        }
        .why-traderiser-text {
            color: #e0e0e0;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .why-traderiser-list {
            text-align: left;
            color: #b0b0b0;
            font-size: 14px;
            margin: 0 auto;
            max-width: 600px;
            list-style-type: none;
            padding: 0;
        }
        .why-traderiser-list li {
            margin: 10px 0;
            position: relative;
            padding-left: 25px;
            transition: color 0.3s ease;
        }
        .why-traderiser-list li:hover {
            color: #ffffff;
        }
        .why-traderiser-list li:before {
            content: "✔";
            color: #00ffcc;
            position: absolute;
            left: 0;
            font-size: 16px;
            animation: checkPulse 2s infinite;
        }
        @keyframes checkPulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.3); }
            100% { transform: scale(1); }
        }
        .popup {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(0, 255, 204, 0.15));
            border-radius: 12px;
            padding: 35px;
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
            text-align: center;
            animation: popupOrbit 0.6s ease-in-out, popupFadeOut 0.6s ease-in-out 1.4s;
            backdrop-filter: blur(10px);
            z-index: 1000;
            width: 90%;
            max-width: 400px;
        }
        @keyframes popupOrbit {
            0% { opacity: 0; transform: translate(-50%, -60%) scale(0.9) rotate(-5deg); }
            100% { opacity: 1; transform: translate(-50%, -50%) scale(1) rotate(0deg); }
        }
        @keyframes popupFadeOut {
            0% { opacity: 1; transform: translate(-50%, -50%) scale(1) rotate(0deg); }
            100% { opacity: 0; transform: translate(-50%, -60%) scale(0.9) rotate(5deg); }
        }
        .popup-title {
            color: #00ffcc;
            font-size: 28px;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
        }
        .popup-text {
            color: #e0e0e0;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .login-button {
            background: #00ffcc;
            color: #1a1a2e;
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.4s ease;
            margin: 5px;
            width: 120px;
        }
        .login-button:hover {
            background: #00ccaa;
            box-shadow: 0 0 20px #00ffcc, inset 0 0 10px #00ffcc;
            transform: scale(1.15) rotate(2deg);
            animation: orbitGlow 0.8s infinite;
        }
        .center-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
        }
        .center-why-traderiser {
            display: flex;
            justify-content: center;
            margin: 25px 0;
        }
        .alert-popup {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 255, 204, 0.25);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
            text-align: center;
            animation: alertOrbit 0.6s ease-in-out, alertFadeOut 0.6s ease-in-out 3.4s;
            z-index: 1000;
            width: 90%;
            max-width: 450px;
        }
        @keyframes alertOrbit {
            0% { opacity: 0; transform: translateX(-50%) translateY(-30px) scale(0.9); }
            100% { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
        }
        @keyframes alertFadeOut {
            0% { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
            100% { opacity: 0; transform: translateX(-50%) translateY(-30px) scale(0.9); }
        }
        .alert-text {
            color: #00ffcc;
            font-size: 16px;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
        }
        .intro-video {
            margin: 30px auto;
            max-width: 700px;
            text-align: center;
        }
        .intro-video iframe {
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .intro-video iframe:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
            animation: orbitGlow 1s infinite;
        }
        @media (max-width: 768px) {
            .login-title {
                font-size: 32px;
            }
            .welcome-text {
                font-size: 16px;
                padding: 0 15px;
            }
            .features-section {
                flex-direction: column;
                align-items: center;
            }
            .feature-card {
                width: 90%;
                max-width: 300px;
            }
            .market-stats {
                flex-direction: column;
                align-items: center;
            }
            .stat-card {
                width: 90%;
                max-width: 200px;
            }
            .testimonials-section {
                padding: 0 10px;
            }
            .testimonial {
                max-width: 100%;
            }
            .why-traderiser-section {
                max-width: 90%;
                padding: 15px;
            }
            .why-traderiser-title {
                font-size: 24px;
            }
            .why-traderiser-text {
                font-size: 14px;
            }
            .why-traderiser-list {
                font-size: 12px;
            }
            .popup {
                width: 90%;
                padding: 20px;
            }
            .popup-title {
                font-size: 24px;
            }
            .popup-text {
                font-size: 14px;
            }
            .login-button {
                width: 100px;
                padding: 8px 16px;
            }
            .alert-popup {
                width: 90%;
                padding: 15px;
            }
            .alert-text {
                font-size: 14px;
            }
            .intro-video {
                max-width: 90%;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Ticker and Particle Effects
    st.markdown("""
        <div class="ticker-bg">
            <div class="ticker">
                AAPL +2.3% • TSLA -1.5% • GOOGL +0.8% • MSFT -0.2% • AMZN +1.7% • NVDA -0.9% • META +1.2% • 
                AMD +3.1% • INTC -0.5% • PYPL +2.0% • AAPL +2.3% • TSLA -1.5% • GOOGL +0.8%
            </div>
        </div>
        <div class="particles">
            <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
            <div class="particle" style="left: 25%; animation-delay: 1s;"></div>
            <div class="particle" style="left: 40%; animation-delay: 2s;"></div>
            <div class="particle" style="left: 55%; animation-delay: 3s;"></div>
            <div class="particle" style="left: 70%; animation-delay: 4s;"></div>
            <div class="particle" style="left: 85%; animation-delay: 5s;"></div>
            <div class="particle" style="left: 15%; animation-delay: 6s;"></div>
            <div class="particle" style="left: 30%; animation-delay: 7s;"></div>
            <div class="particle" style="left: 45%; animation-delay: 8s;"></div>
        </div>
    """, unsafe_allow_html=True)

    # Show popup for 2 seconds only once per session
    current_time = time.time()
    if st.session_state.show_popup and (current_time - st.session_state.popup_start_time) <= 2:
        with st.container():
            st.markdown("""
                <div class="popup">
                    <h2 class="popup-title">🎉 Welcome to TradeRiser!</h2>
                    <p class="popup-text">Register now for a $10,000 bonus and unlock your trading potential! 💰</p>
                </div>
            """, unsafe_allow_html=True)
    elif st.session_state.show_popup and (current_time - st.session_state.popup_start_time) > 2:
        st.session_state.show_popup = False

    st.markdown('<h1 class="login-title">🚀 TradeRiser :</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="login-title"> Rise With Every Trade</h3>', unsafe_allow_html=True)
    st.markdown("""
        <div class="welcome-text">
            Step into the world of trading with TradeRiser! Experience real-time stock trading, portfolio management, and cutting-edge market insights. Join thousands of traders and start your journey to financial success today! 📈
        </div>
    """, unsafe_allow_html=True)

    # Additional Content: Market Stats
    st.markdown("""
        <div class="market-stats">
            <div class="stat-card">
                <div class="stat-title">S&P 500</div>
                <div class="stat-value">4,320.45 <span style="color: #00ff00;">+0.8%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Dow Jones</div>
                <div class="stat-value">34,567.89 <span style="color: #ff0000;">-0.3%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-title">NASDAQ</div>
                <div class="stat-value">14,123.45 <span style="color: #00ff00;">+1.2%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Bitcoin</div>
                <div class="stat-value">$65,432 <span style="color: #00ff00;">+2.5%</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Login Form
    username = st.text_input("Username 👤", key="login_username").strip()
    password = st.text_input("Password 🔒", type="password", key="login_password")
    st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
    login_btn = st.button("Login 🔐", key="login_btn")
    register_btn = st.button("Register 📝", key="register_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if login_btn:
        users = st.session_state.users
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.email = users[username]["email"]
            st.session_state.price_alerts = users[username].get("price_alerts", [])
            st.success("Welcome back, trader! 🚀")
            st.rerun()
        else:
            st.error("Invalid credentials! Try again. 🚫")
    if register_btn:
        st.session_state.show_register = True
        st.rerun()

    # Additional Content: Features Section
    st.markdown("""
        <div class="features-section">
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Real-Time Trading</div>
                <div class="feature-desc">Access live market data instantly.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💼</div>
                <div class="feature-title">Portfolio Tracking</div>
                <div class="feature-desc">Monitor your investments effortlessly.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Market Insights</div>
                <div class="feature-desc">Make informed decisions with data.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">👥</div>
                <div class="feature-title">Trading Community</div>
                <div class="feature-desc">Connect with traders worldwide.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔔</div>
                <div class="feature-title">Price Alerts</div>
                <div class="feature-desc">Stay updated with custom alerts.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI Tools</div>
                <div class="feature-desc">Leverage AI for smarter trades.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Additional Content: Intro Video
    st.markdown("""
        <div class="intro-video">
            <h3 style="color: #00ffcc; text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);">What our users say!</h3>
        </div>
    """, unsafe_allow_html=True)

    # Testimonials
    st.markdown('<div class="testimonials-section">', unsafe_allow_html=True)
    st.markdown("""
        <div class="testimonial">
            <p class="testimonial-text">"TradeRiser transformed my trading game—up 30% in just 3 months!"</p>
            <p class="testimonial-author">- Sarah K., Pro Trader</p>
        </div>
        <div class="testimonial">
            <p class="testimonial-text">"Real-time data and a vibrant community—what more could a trader ask for?"</p>
            <p class="testimonial-author">- Michael T., Investor</p>
        </div>
        <div class="testimonial">
            <p class="testimonial-text">"The AI tools gave me an edge I never knew I needed!"</p>
            <p class="testimonial-author">- Priya R., Day Trader</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Why TradeRiser Button
    st.markdown('<div class="center-why-traderiser">', unsafe_allow_html=True)
    if st.button("Why TradeRiser? 🔍", key="why_traderiser_btn"):
        st.session_state.show_why_traderiser = not st.session_state.show_why_traderiser
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_why_traderiser:
        st.markdown("""
            <div class="why-traderiser-section">
                <h2 class="why-traderiser-title">Why Choose TradeRiser?</h2>
                <p class="why-traderiser-text">
                    TradeRiser is your ultimate partner in mastering the stock market with innovative tools and a supportive community. 🚀
                </p>
                <ul class="why-traderiser-list">
                    <li><strong>Real-Time Data:</strong> Stay ahead with live updates.</li>
                    <li><strong>Intuitive Design:</strong> Easy for beginners, powerful for pros.</li>
                    <li><strong>Advanced Analytics:</strong> Predict trends with precision.</li>
                    <li><strong>Community Power:</strong> Learn and grow with fellow traders.</li>
                    <li><strong>Top Security:</strong> Your data and funds are safe.</li>
                    <li><strong>24/7 Support:</strong> Expert help whenever you need it.</li>
                    <li><strong>AI Assistance:</strong> Smart tools for smarter trades.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Register Page
def register():
    st.markdown("""
        <style>
        .register-container {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 35px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            text-align: center;
            animation: techyBounce 0.8s ease-in-out;
            max-width: 400px;
            margin: 80px auto;
            backdrop-filter: blur(10px);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .register-container:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
            animation: orbitGlow 1s infinite;
        }
        @media (max-width: 768px) {
            .register-container {
                max-width: 90%;
                padding: 20px;
                margin: 40px auto;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="register-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">Register 📝</h1>', unsafe_allow_html=True)
    username = st.text_input("New Username 👤", key="register_username")
    email = st.text_input("Email 📧", key="register_email")
    password = st.text_input("New Password 🔒", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password 🔑", type="password", key="register_confirm_password")
    st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
    register_btn = st.button("Register 📋", key="register_submit_btn")
    back_btn = st.button("Back to Login 🔙", key="back_to_login_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if register_btn:
        users = st.session_state.users
        if username in users:
            st.error("Username already exists! 🚫")
        elif password != confirm_password:
            st.error("Passwords do not match! ⚠️")
        elif not email:
            st.error("Please provide an email address! 📧")
        else:
            users[username] = {
                "password": password,
                "email": email,
                "balance": 10000.0,
                "portfolio": {},
                "watchlist": [],
                "transactions": [],
                "price_alerts": []
            }
            save_users(users)
            st.session_state.users = users
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.email = email
            st.session_state.price_alerts = []
            # Send welcome email
            subject = "Welcome to TradeRiser! 🎉"
            body = f"""
🌟 Hello {username}, 🌟

🎉 Welcome to TradeRiser! 🎉

   You're now part of our trading family! We've added a $10,000 bonus to your account to get you started.

💡 Happy Trading!
The TradeRiser Team 🌟
"""
            if send_email(email, subject, body):
                st.success(f"Welcome, {username}! $10,000 bonus added. Check your email for a welcome message! 💰")
            else:
                st.warning(f"Welcome, {username}! $10,000 bonus added. Failed to send welcome email. 💰")
            st.rerun()
    if back_btn:
        st.session_state.show_register = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Logout
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.trading_active = False
    st.session_state.candle_data = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    st.session_state.last_update_time = 0
    st.session_state.price_alerts = []
    st.session_state.alert_popup = False
    st.session_state.alert_message = ""
    st.session_state.bought_price = {}
    st.session_state.sold_price = {}
    st.session_state.current_price = 0.0
    st.session_state.buy_message = ""
    st.session_state.sell_message = ""
    st.session_state.show_popup = True
    st.session_state.watchlist_last_update = 0
    st.success("Logged out successfully! 👋")
    st.rerun()

# Calculate portfolio stats
def calculate_portfolio_stats(user_data):
    portfolio_value = 0.0
    total_shares = 0
    total_assets = 0
    net_profit_loss = 0.0
    breakdown = []

    for symbol, details in user_data.get("portfolio", {}).items():
        current_price = st.session_state.last_price.get(symbol, get_current_price(symbol))
        asset_value = round(current_price * details["quantity"], 2)
        portfolio_value += asset_value
        total_shares += details["quantity"]
        total_assets += 1
        asset_profit_loss = round((current_price - details["avg_price"]) * details["quantity"], 2)
        net_profit_loss += asset_profit_loss
        breakdown.append({
            "Symbol": symbol,
            "Quantity": details["quantity"],
            "Avg price": details["avg_price"],
            "Current price": current_price,
            "Value": asset_value,
            "Profit/loss": asset_profit_loss
        })

    st.session_state.portfolio_history.append({"time": datetime.now(), "value": portfolio_value + user_data["balance"]})
    if len(st.session_state.portfolio_history) > 50:
        st.session_state.portfolio_history = st.session_state.portfolio_history[-50:]

    return {
        "portfolio_value": portfolio_value,
        "cash_balance": user_data["balance"],
        "total_shares": total_shares,
        "total_assets": total_assets,
        "net_profit_loss": net_profit_loss,
        "breakdown": breakdown
    }

# Main App
def main_app():
    users = st.session_state.users
    user_data = users.get(st.session_state.username, {})
    if not user_data:
        st.error("User data not found. Please log in again.")
        return

    st.markdown("""
        <style>
        .glass {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            backdrop-filter: blur(10px);
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
            animation: fadeIn 0.6s ease-in-out;
            border: 1px solid rgba(0, 255, 204, 0.2);
            margin-left: auto;
            margin-right: auto;
            max-width: 1200px;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .glass h3 {
            color: #00ffcc;
            text-shadow: 0 0 6px rgba(0, 255, 255, 0.6);
            font-size: 20px;
            text-align: center;
        }
        .glass p {
            color: #e0e0e0;
            font-size: 16px;
        }
        .glass button {
            padding: 8px 16px;
            border-radius: 25px;
            border: none;
            cursor: pointer;
            margin: 5px;
            transition: all 0.4s ease;
        }
        .glass button:hover {
            transform: scale(1.15) rotate(3deg);
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.6), inset 0 0 5px rgba(0, 255, 255, 0.3);
            animation: orbitGlow 0.8s infinite;
        }
        .buy-btn { background: #00ff00; color: #000; }
        .sell-btn { background: #ff0000; color: #fff; }
        .trade-message {
            margin-top: 10px;
            font-weight: bold;
            font-size: 16px;
            animation: slideIn 0.6s ease-in-out;
            text-align: center;
        }
        @keyframes slideIn {
            0% { opacity: 0; transform: translateX(-30px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        .premium-section {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(0, 255, 204, 0.15));
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.4);
            backdrop-filter: blur(10px);
            animation: fadeIn 0.6s ease-in-out;
            margin-left: auto;
            margin-right: auto;
            max-width: 1200px;
        }
        .premium-title {
            color: #ff00ff;
            font-size: 28px;
            font-weight: bold;
            text-shadow: 0 0 12px rgba(255, 0, 255, 0.6);
            margin-bottom: 20px;
            text-align: center;
        }
        .premium-subtitle {
            color: #e0e0e0;
            font-size: 18px;
            margin-bottom: 25px;
            text-align: center;
        }
        .premium-features {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }
        .premium-feature-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 20px;
            width: 220px;
            text-align: center;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            animation: techyBounce 1.2s ease-in-out;
            border: 1px solid rgba(255, 0, 255, 0.2);
        }
        .premium-feature-card:hover {
            transform: translateY(-10px) rotate(2deg);
            box-shadow: 0 0 25px rgba(255, 0, 255, 0.5), inset 0 0 10px rgba(255, 0, 255, 0.3);
            animation: orbitGlow 1s infinite;
        }
        .premium-feature-icon {
            font-size: 34px;
            color: #ff00ff;
            margin-bottom: 12px;
            animation: spinPulse 3s infinite ease-in-out;
        }
        .premium-feature-title {
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
        }
        .premium-feature-desc {
            color: #b0b0b0;
            font-size: 14px;
        }
        .premium-more {
            color: #00ffcc;
            font-size: 16px;
            font-style: italic;
            margin-top: 25px;
            text-align: center;
        }
        .join-waitlist-btn {
            background: #ff00ff;
            color: #ffffff;
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.4s ease;
            margin: 10px auto;
            display: block;
            width: 150px;
        }
        .join-waitlist-btn:hover {
            background: #cc00cc;
            box-shadow: 0 0 20px #ff00ff, inset 0 0 10px #ff00ff;
            transform: scale(1.15) rotate(2deg);
            animation: orbitGlow 0.8s infinite;
        }
        .sidebar-watchlist {
            margin-top: 25px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            animation: fadeIn 0.6s ease-in-out;
            border: 1px solid rgba(0, 255, 204, 0.2);
        }
        .sidebar-watchlist h4 {
            color: #00ffcc;
            font-size: 18px;
            margin-bottom: 12px;
            text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
            text-align: center;
        }
        .sidebar-watchlist p {
            color: #e0e0e0;
            font-size: 16px;
            margin: 8px 0;
            text-align: center;
        }
        .dashboard-info {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-left: auto;
            margin-right: auto;
            max-width: 1000px;
        }
        .info-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 15px;
            width: 200px;
            text-align: center;
            animation: techyBounce 1s ease-in-out;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            border: 1px solid rgba(0, 255, 204, 0.2);
        }
        .info-card:hover {
            transform: translateY(-10px) scale(1.05);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
            animation: orbitGlow 1s infinite;
        }
        .info-title {
            color: #00ffcc;
            font-size: 16px;
            margin-bottom: 8px;
        }
        .info-value {
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .glass {
                max-width: 90%;
                padding: 15px;
            }
            .glass h3 {
                font-size: 18px;
            }
            .glass p {
                font-size: 14px;
            }
            .trade-message {
                font-size: 14px;
            }
            .premium-section {
                max-width: 90%;
                padding: 15px;
            }
            .premium-title {
                font-size: 24px;
            }
            .premium-subtitle {
                font-size: 16px;
            }
            .premium-features {
                flex-direction: column;
                align-items: center;
            }
            .premium-feature-card {
                width: 90%;
                max-width: 300px;
            }
            .premium-more {
                font-size: 14px;
            }
            .join-waitlist-btn {
                width: 120px;
                padding: 8px 16px;
            }
            .sidebar-watchlist {
                padding: 10px;
            }
            .sidebar-watchlist h4 {
                font-size: 16px;
            }
            .sidebar-watchlist p {
                font-size: 14px;
            }
            .dashboard-info {
                flex-direction: column;
                align-items: center;
            }
            .info-card {
                width: 90%;
                max-width: 250px;
            }
            .info-title {
                font-size: 14px;
            }
            .info-value {
                font-size: 16px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Preload data at startup
    preload_data()

    # Show alert popup if there’s an alert
    if st.session_state.alert_popup:
        st.markdown(f"""
            <div class="alert-popup">
                <p class="alert-text">{st.session_state.alert_message}</p>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.alert_popup = False
        st.session_state.alert_message = ""

    # Sidebar with Watchlist
    st.sidebar.title(f"Welcome, {st.session_state.username} 👋")
    portfolio_stats = calculate_portfolio_stats(user_data)
    st.sidebar.write(f"Cash: ${portfolio_stats['cash_balance']:.2f} 💵")
    st.sidebar.write(f"Portfolio: ${portfolio_stats['portfolio_value']:.2f} 📈")

    if user_data.get("watchlist", []):
        st.sidebar.markdown('<div class="sidebar-watchlist"><h4>Watchlist 👀</h4>', unsafe_allow_html=True)
        watchlist_data = fetch_watchlist_data(user_data["watchlist"])
        for _, row in watchlist_data.iterrows():
            st.sidebar.markdown(
                f"<p>{row['Ticker symbol']}: ${row['Price']:.2f}</p>",
                unsafe_allow_html=True
            )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

    menu = [
        "Dashboard 📊", "Portfolio 💼", "Watchlist 👀", "Transactions 📜",
        "Profile Settings 🔧", "Market News 📰", "Market Movers 📊",
        "Learning Resources 📖", "TradeRiser Premium 💎", "Risk Calculator ⚖️",
        "Recent Data 📈", "Price Alerts 🔔"
    ]
    choice = st.sidebar.selectbox("Menu", menu)
    if st.sidebar.button("Logout 🚪"):
        logout()

    # Check Price Alerts
    check_price_alerts()

    # Dashboard
    if choice == "Dashboard 📊":
        symbol = st.text_input("Ticker (e.g., AAPL) 🎫", value=st.session_state.symbol).upper()
        st.session_state.symbol = symbol
        company_name = get_company_name(symbol)
        st.title(f"Dashboard: {symbol} ({company_name}) 📈")

        # Additional Content: Quick Info Cards
        current_price = st.session_state.current_price if st.session_state.current_price > 0 else get_current_price(symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        st.markdown("""
            <div class="dashboard-info">
                <div class="info-card">
                    <div class="info-title">Current price</div>
                    <div class="info-value">${:.2f}</div>
                </div>
                <div class="info-card">
                    <div class="info-title">52w high</div>
                    <div class="info-value">${:.2f}</div>
                </div>
                <div class="info-card">
                    <div class="info-title">52w low</div>
                    <div class="info-value">${:.2f}</div>
                </div>
                <div class="info-card">
                    <div class="info-title">Market cap</div>
                    <div class="info-value">${:.2f}B</div>
                </div>
            </div>
        """.format(
            current_price,
            info.get("fiftyTwoWeekHigh", 0),
            info.get("fiftyTwoWeekLow", 0),
            info.get("marketCap", 0) / 1e9
        ), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            start_trading = st.button("Start Trading 🚀")
        with col2:
            stop_trading = st.button("Stop Trading 🛑")

        if start_trading:
            st.session_state.trading_active = True
            data = get_stock_data(symbol)
            st.session_state.candle_data = data[-15:]
            st.session_state.last_price[symbol] = st.session_state.candle_data.iloc[-1]["close"]
            st.session_state.current_price = st.session_state.last_price[symbol]
            st.session_state.last_update_time = time.time()

        if stop_trading:
            st.session_state.trading_active = False

        # Trade Summary
        bought_price = st.session_state.bought_price.get(symbol, 0.0)
        sold_price = st.session_state.sold_price.get(symbol, 0.0)
        profit_loss = (sold_price - bought_price) if sold_price > 0 else 0.0
        available_stocks = user_data["portfolio"].get(symbol, {"quantity": 0})["quantity"]
        st.markdown(f"""
            <div class="glass">
                <h3>Trade summary 💰</h3>
                <p>Bought price 🛒: ${bought_price:.2f}</p>
                <p>Current price 📊: ${current_price:.2f}</p>
                <p>Sold price 🏷️: ${sold_price:.2f}</p>
                <p>Profit/loss 📈: <span style="color: {'#00ff00' if profit_loss >= 0 else '#ff0000'}">${profit_loss:.2f}</span></p>
                <h4>Available stocks 📜</h4>
                <p>{symbol}: {available_stocks} shares</p>
            </div>
        """, unsafe_allow_html=True)

        # Buy/Sell
        st.markdown('<div class="glass"><h3>Trade 🛠️</h3>', unsafe_allow_html=True)
        quantity = st.number_input("Quantity 🔢", min_value=1, value=1, key="trade_qty")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Buy 🟢", key="buy_btn"):
                price = st.session_state.current_price if st.session_state.current_price > 0 else get_current_price(symbol)
                if price <= 0:
                    st.error("Cannot buy: Current price is zero! 🚫")
                else:
                    total_cost = price * quantity
                    if total_cost <= user_data["balance"]:
                        user_data["balance"] -= total_cost
                        if symbol in user_data["portfolio"]:
                            current_qty = user_data["portfolio"][symbol]["quantity"]
                            current_avg = user_data["portfolio"][symbol]["avg_price"]
                            user_data["portfolio"][symbol]["quantity"] += quantity
                            user_data["portfolio"][symbol]["avg_price"] = (
                                (current_avg * current_qty + price * quantity) / (current_qty + quantity)
                            )
                        else:
                            user_data["portfolio"][symbol] = {"quantity": quantity, "avg_price": price}
                        user_data["transactions"].append({
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol, "action": "Buy", "quantity": quantity, "price": price, "total": total_cost
                        })
                        st.session_state.bought_price[symbol] = price
                        st.session_state.current_price = price
                        st.session_state.buy_message = f"Bought {quantity} shares at ${price:.2f}"
                        save_users(users)
                    else:
                        st.error("Insufficient funds! 🚫")
        with col2:
            if st.button("Sell 🔴", key="sell_btn"):
                price = st.session_state.current_price if st.session_state.current_price > 0 else get_current_price(symbol)
                if price <= 0:
                    st.error("Cannot sell: Current price is zero! 🚫")
                else:
                    if symbol in user_data["portfolio"] and user_data["portfolio"][symbol]["quantity"] >= quantity:
                        total_cost = price * quantity
                        user_data["portfolio"][symbol]["quantity"] -= quantity
                        user_data["balance"] += total_cost
                        if user_data["portfolio"][symbol]["quantity"] == 0:
                            del user_data["portfolio"][symbol]
                        user_data["transactions"].append({
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol, "action": "Sell", "quantity": quantity, "price": price, "total": total_cost
                        })
                        st.session_state.sold_price[symbol] = price
                        st.session_state.current_price = price
                        st.session_state.sell_message = f"Sold {quantity} shares at ${price:.2f}"
                        save_users(users)
                    else:
                        st.error("Not enough shares! 🚫")

        # Display Buy/Sell Messages
        if st.session_state.buy_message:
            st.markdown(f'<p class="trade-message" style="color: #00ff00;">{st.session_state.buy_message}</p>', unsafe_allow_html=True)
        if st.session_state.sell_message:
            st.markdown(f'<p class="trade-message" style="color: #ff0000;">{st.session_state.sell_message}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Candlestick chart with smooth updates every second
        if not st.session_state.candle_data.empty:
            chart_placeholder = st.empty()
            
            # Initial chart rendering
            fig = go.Figure(data=[go.Candlestick(
                x=st.session_state.candle_data["time"],
                open=st.session_state.candle_data["open"],
                high=st.session_state.candle_data["high"],
                low=st.session_state.candle_data["low"],
                close=st.session_state.candle_data["close"],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.update_layout(
                title=f"{symbol} Candlestick Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                height=500,
                template="plotly_dark",
                xaxis=dict(tickformat="%H:%M:%S", tickangle=45, nticks=8),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Smooth update loop every second when trading is active
            if st.session_state.trading_active:
                while st.session_state.trading_active:
                    # Update candle data every second
                    update_candle_data(symbol)
                    # Update the chart with new data
                    fig = go.Figure(data=[go.Candlestick(
                        x=st.session_state.candle_data["time"],
                        open=st.session_state.candle_data["open"],
                        high=st.session_state.candle_data["high"],
                        low=st.session_state.candle_data["low"],
                        close=st.session_state.candle_data["close"],
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    )])
                    fig.update_layout(
                        title=f"{symbol} Candlestick Chart",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=True,
                        height=500,
                        template="plotly_dark",
                        xaxis=dict(tickformat="%H:%M:%S", tickangle=45, nticks=8),
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    # Wait for 1 second before the next update
                    time.sleep(1)

    # Portfolio
    elif choice == "Portfolio 💼":
        st.title("Portfolio 💼")
        stats = calculate_portfolio_stats(user_data)
        st.markdown(f"""
            <div class="glass">
                <h3>Account summary</h3>
                <p>Cash balance: ${stats['cash_balance']:.2f}</p>
                <p>Portfolio value: ${stats['portfolio_value']:.2f}</p>
                <p>Total assets: {stats['total_assets']}</p>
                <p>Total shares: {stats['total_shares']}</p>
                <p>Net p/l: <span style="color: {'#00ff00' if stats['net_profit_loss'] >= 0 else '#ff0000'}">${stats['net_profit_loss']:.2f}</span></p>
            </div>
        """, unsafe_allow_html=True)

        if stats["breakdown"]:
            st.markdown('<div class="glass"><h3>Holdings breakdown</h3>', unsafe_allow_html=True)
            breakdown_df = pd.DataFrame(stats["breakdown"])
            breakdown_df.index = range(1, len(breakdown_df) + 1)
            st.table(breakdown_df)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.portfolio_history:
                df = pd.DataFrame(st.session_state.portfolio_history)
                fig = px.line(df, x="time", y="value", title="Portfolio Performance Over Time", template="plotly_dark")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Watchlist
    elif choice == "Watchlist 👀":
        st.title("Watchlist 👀")
        new_symbol = st.text_input("Add Symbol ➕").upper()
        if st.button("Add ➕"):
            if new_symbol and new_symbol not in user_data["watchlist"]:
                user_data["watchlist"].append(new_symbol)
                save_users(users)
                st.success(f"{new_symbol} added to watchlist! ✅")

        if user_data["watchlist"]:
            st.markdown('<div class="glass"><h3>Your watchlist</h3>', unsafe_allow_html=True)
            watchlist_data = fetch_watchlist_data(user_data["watchlist"])
            st.table(watchlist_data)
            st.markdown('</div>', unsafe_allow_html=True)

    # Transactions
    elif choice == "Transactions 📜":
        st.title("Transactions 📜")
        if user_data["transactions"]:
            st.markdown('<div class="glass"><h3>Your transactions</h3>', unsafe_allow_html=True)
            transactions_df = pd.DataFrame(user_data["transactions"])
            transactions_df.index = range(1, len(transactions_df) + 1)
            # Convert column names to sentence case
            transactions_df.columns = [col.capitalize() if col.lower() == col else ' '.join(word.capitalize() if i == 0 else word.lower() for i, word in enumerate(col.split())) for col in transactions_df.columns]
            st.table(transactions_df)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass"><h3>No transactions yet</h3><p>Start trading to see your transactions here! 🚀</p></div>', unsafe_allow_html=True)

    # Profile Settings
    elif choice == "Profile Settings 🔧":
        st.title("Profile Settings 🔧")
        st.markdown('<div class="glass"><h3>Update your profile</h3>', unsafe_allow_html=True)
        new_email = st.text_input("Update Email 📧", value=user_data["email"])
        new_password = st.text_input("Update Password 🔒", type="password", value="")
        confirm_password = st.text_input("Confirm New Password 🔑", type="password", value="")
        if st.button("Update Profile 🔄"):
            if new_password and new_password != confirm_password:
                st.error("Passwords do not match! ⚠️")
            else:
                if new_email:
                    user_data["email"] = new_email
                    st.session_state.email = new_email
                if new_password:
                    user_data["password"] = new_password
                save_users(users)
                st.success("Profile updated successfully! ✅")
        st.markdown('</div>', unsafe_allow_html=True)

    # Market News
    elif choice == "Market News 📰":
        st.title("Market News 📰")
        st.markdown('<div class="glass"><h3>Latest market news</h3>', unsafe_allow_html=True)
        current_time = time.time()
        if current_time - st.session_state.news_last_update >= 5:
            st.session_state.market_news = fetch_market_news()
            st.session_state.news_last_update = current_time
        for news in st.session_state.market_news:
            st.markdown(f"<p>{news}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Market Movers
    elif choice == "Market Movers 📊":
        st.title("Market Movers 📊")
        st.markdown('<div class="glass"><h3>Top gainers</h3>', unsafe_allow_html=True)
        for mover in st.session_state.market_movers["gainers"]:
            st.markdown(f"<p>{mover['symbol']}: +{mover['change']}%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Learning Resources
    elif choice == "Learning Resources 📖":
        st.title("Learning Resources 📖")
        st.markdown('<div class="glass"><h3>Explore trading guides</h3>', unsafe_allow_html=True)
        learning_options = [
            "- [Understanding Market Trends](#)",
            "- [How to Use TradeRiser Effectively](#)",
            "- [Advanced Trading Strategies](#)",
            "- [Introduction to Options Trading](#)",
            "- [Fundamental Analysis for Beginners](#)",
            "- [Day Trading Tips and Tricks](#)"
        ]
        current_time = time.time()
        if current_time - st.session_state.learning_last_update >= 5:
            st.session_state.learning_last_update = current_time
        for resource in learning_options:
            st.write(resource)
        st.markdown('</div>', unsafe_allow_html=True)

    # TradeRiser Premium
    elif choice == "TradeRiser Premium 💎":
        st.title("TradeRiser Premium 💎")
        st.markdown("""
            <div class="premium-section">
                <h2 class="premium-title">Unlock TradeRiser Premium 💎</h2>
                <p class="premium-subtitle">Elevate your trading with exclusive features and insights!</p>
                <div class="premium-features">
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">📈</div>
                        <div class="premium-feature-title">Advanced analytics</div>
                        <div class="premium-feature-desc">Predict trends with AI-powered tools.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">🔔</div>
                        <div class="premium-feature-title">Priority alerts</div>
                        <div class="premium-feature-desc">Get instant notifications on market moves.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">📚</div>
                        <div class="premium-feature-title">Exclusive resources</div>
                        <div class="premium-feature-desc">Access premium trading guides.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">🤝</div>
                        <div class="premium-feature-title">VIP support</div>
                        <div class="premium-feature-desc">24/7 priority customer support.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">🎨</div>
                        <div class="premium-feature-title">Customizable dashboards</div>
                        <div class="premium-feature-desc">Tailor your trading interface.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">🔍</div>
                        <div class="premium-feature-title">Real-time market scanner</div>
                        <div class="premium-feature-desc">Identify opportunities with live scans.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">🤖</div>
                        <div class="premium-feature-title">Automated trading bots</div>
                        <div class="premium-feature-desc">Execute trades with smart bots.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">📅</div>
                        <div class="premium-feature-title">Extended historical data</div>
                        <div class="premium-feature-desc">Access up to 10 years of data.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">📉</div>
                        <div class="premium-feature-title">Portfolio optimization</div>
                        <div class="premium-feature-desc">Optimize investments with algorithms.</div>
                    </div>
                    <div class="premium-feature-card">
                        <div class="premium-feature-icon">🎥</div>
                        <div class="premium-feature-title">Exclusive webinars</div>
                        <div class="premium-feature-desc">Join live sessions with experts.</div>
                    </div>
                </div>
                <p class="premium-more">...and much more! 🚀</p>
        """, unsafe_allow_html=True)

        # Join Waitlist Button
        if st.button("Join Waitlist 🚀", key="join_waitlist_btn", help="Join the waitlist to get early access to TradeRiser Premium!"):
            user_email = st.session_state.email
            if user_email:
                # Prepare the email content with all 10 features
                subject = "TradeRiser Premium Waitlist Confirmation 🎉"
                body = f"""
🌟 Hello {st.session_state.username}, 🌟

🎉 Thank you for joining the TradeRiser Premium waitlist! You're one step closer to unlocking exclusive features that will take your trading to the next level! 🚀

Here’s what you’ll get with TradeRiser Premium:

1. 📈 **Advanced Analytics** - Predict trends with AI-powered tools.
2. 🔔 **Priority Alerts** - Get instant notifications on market moves.
3. 📚 **Exclusive Resources** - Access premium trading guides and tutorials.
4. 🤝 **VIP Support** - 24/7 priority customer support.
5. 🎨 **Customizable Dashboards** - Tailor your trading interface to your preferences.
6. 🔍 **Real-Time Market Scanner** - Identify opportunities with live market scans.
7. 🤖 **Automated Trading Bots** - Set up bots to execute trades based on your strategies.
8. 📅 **Extended Historical Data** - Access up to 10 years of historical market data.
9. 📉 **Portfolio Optimization Tools** - Optimize your investments with advanced algorithms.
10. 🎥 **Exclusive Webinars** - Join live sessions with top trading experts.

💡 We’ll notify you as soon as TradeRiser Premium is available for you! Stay tuned for more updates.

Happy Trading! 😊
The TradeRiser Team 🌟
"""
                if send_email(user_email, subject, body):
                    st.success(f"You’ve joined the TradeRiser Premium waitlist! A confirmation email has been sent to {user_email}. 🎉")
                else:
                    st.warning("You’ve joined the waitlist, but we couldn’t send the confirmation email. Please check your email settings. 📧")
            else:
                st.error("No email found for your account. Please update your email in Profile Settings. 📧")

        st.markdown('</div>', unsafe_allow_html=True)

    # Risk Calculator
    elif choice == "Risk Calculator ⚖️":
        st.title("Risk Calculator ⚖️")
        st.markdown('<div class="glass"><h3>Calculate your risk</h3>', unsafe_allow_html=True)
        entry_price = st.number_input("Entry price 💵", min_value=0.0, value=100.0)
        stop_loss = st.number_input("Stop loss 🛑", min_value=0.0, value=90.0)
        position_size = st.number_input("Position size (shares) 📊", min_value=1, value=100)
        if entry_price > 0 and stop_loss >= 0 and position_size > 0:
            risk_per_share = entry_price - stop_loss
            total_risk = risk_per_share * position_size
            st.markdown(f"""
                <p>Risk per share: ${risk_per_share:.2f}</p>
                <p>Total risk: ${total_risk:.2f}</p>
                <p style="color: {'#ff0000' if total_risk > user_data['balance'] * 0.02 else '#00ff00'}">
                    Risk level: { 'High (exceeds 2% of balance)' if total_risk > user_data['balance'] * 0.02 else 'Acceptable'}
                </p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent Data
    elif choice == "Recent Data 📈":
        st.title("Recent Data 📈")
        st.markdown('<div class="glass"><h3>Market overview</h3>', unsafe_allow_html=True)
        recent_data = st.session_state.recent_data
        # Convert column names to sentence case
        recent_data.columns = [col.capitalize() if col.lower() == col else ' '.join(word.capitalize() if i == 0 else word.lower() for i, word in enumerate(col.split())) for col in recent_data.columns]
        st.table(recent_data)
        st.markdown('</div>', unsafe_allow_html=True)

    # Price Alerts
    elif choice == "Price Alerts 🔔":
        st.title("Price Alerts 🔔")
        st.markdown('<div class="glass"><h3>Set price alerts</h3>', unsafe_allow_html=True)
        alert_symbol = st.text_input("Symbol 🎫", value="AAPL").upper()
        target_price = st.number_input("Target price 🎯", min_value=0.0, value=150.0)
        if st.button("Set Alert 🔔"):
            st.session_state.price_alerts.append({"symbol": alert_symbol, "target_price": target_price})
            user_data["price_alerts"] = st.session_state.price_alerts
            save_users(users)
            st.success(f"Alert set for {alert_symbol} at ${target_price:.2f}! 🔔")

        if st.session_state.price_alerts:
            st.markdown('<h4>Your alerts</h4>', unsafe_allow_html=True)
            alerts_df = pd.DataFrame(st.session_state.price_alerts)
            alerts_df["Current price"] = alerts_df["symbol"].apply(lambda x: st.session_state.last_price.get(x, get_current_price(x)))
            alerts_df.index = range(1, len(alerts_df) + 1)
            # Convert column names to sentence case
            alerts_df.columns = [col.capitalize() if col.lower() == col else ' '.join(word.capitalize() if i == 0 else word.lower() for i, word in enumerate(col.split())) for col in alerts_df.columns]
            st.table(alerts_df)
        st.markdown('</div>', unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    st.set_page_config(page_title="TradeRiser", page_icon="📈", layout="wide")
    initialize_session_state()

    if not st.session_state.logged_in:
        if st.session_state.show_register:
            register()
        else:
            login()
    else:
        main_app()