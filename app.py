"""
Main Streamlit Application
"""

import streamlit as st
import matplotlib.pyplot as plt

from data import load_stock_data
from model import train_model

st.set_page_config(
    page_title="AI Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Market Prediction Dashboard")

st.write(
    """
Predict tomorrow's closing stock price using historical stock market data
and Machine Learning.
"""
)

ticker = st.selectbox(
    "Choose Stock",
    [
        "AAPL",
        "TSLA",
        "MSFT",
        "GOOGL",
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS"
    ]
)

if st.button("🚀 Predict Price"):

    with st.spinner("Fetching latest stock market data..."):
        df = load_stock_data(ticker)

    (
        model,
        X_test,
        y_test,
        predictions,
        r2,
        mae,
        rmse
    ) = train_model(df)

    # Latest available features
    latest_features = df[[
        "Open",
        "High",
        "Low",
        "Volume",
        "Previous Close",
        "MA5",
        "MA10",
        "MA20",
        "Daily Return",
        "Price Range"
    ]].iloc[[-1]]

    # Predict next day's closing price
    next_day_prediction = float(model.predict(latest_features)[0])

    # Today's closing price
    today_close = float(df["Close"].values[-1])

    # Change calculations
    price_change = next_day_prediction - today_close
    percentage_change = (price_change / today_close) * 100

    st.success("✅ Prediction Generated Successfully")

    st.subheader("📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Today's Close",
        f"${today_close:.2f}"
    )

    col2.metric(
        "Tomorrow Prediction",
        f"${next_day_prediction:.2f}",
        f"{price_change:+.2f}"
    )

    trend = "🟢 Bullish" if price_change >= 0 else "🔴 Bearish"

    col3.metric(
        "Expected Change",
        f"{percentage_change:.2f}%"
    )

    st.write(f"### Market Trend: {trend}")

    st.subheader("📈 Model Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric("R² Score", f"{r2:.3f}")
    m2.metric("MAE", f"{mae:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")

    st.subheader("📉 Actual vs Predicted Closing Price")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        y_test.values,
        label="Actual",
        linewidth=2
    )

    ax.plot(
        predictions,
        label="Predicted",
        linewidth=2,
        linestyle="--"
    )

    ax.set_xlabel("Days")
    ax.set_ylabel("Closing Price")
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.legend()

    st.pyplot(fig)

    st.subheader("📋 Complete Stock Dataset")

    st.write(f"Total Records: {len(df)}")

    st.dataframe(
       df,
       use_container_width=True,
       height=500
    )
