"""
Main Streamlit Application
"""

import streamlit as st
import matplotlib.pyplot as plt

from data import load_stock_data
from model import train_model

st.set_page_config(
    page_title="Stock Market Predictor",
    layout="wide"
)

st.title("📈 Stock Market Predictor")

st.write(
    """
Predict the **next day's closing stock price**
using Machine Learning.
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

if st.button("Predict"):

    with st.spinner("Downloading stock data..."):

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

    next_day_prediction = model.predict(
        [X_test.iloc[-1]]
    )[0]

    st.success("Prediction Complete")

    st.subheader("Predicted Next-Day Closing Price")

    st.metric(
        "Closing Price",
        f"${next_day_prediction:.2f}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        y_test.values,
        label="Actual"
    )

    ax.plot(
        predictions,
        label="Predicted"
    )

    ax.legend()

    ax.set_xlabel("Days")
    ax.set_ylabel("Closing Price")

    st.pyplot(fig)

    st.subheader("Recent Data")

    st.dataframe(df.tail(10))