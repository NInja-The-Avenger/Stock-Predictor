"""
data.py

Downloads historical stock data from Yahoo Finance
and prepares it for machine learning.
"""

import yfinance as yf
import pandas as pd


def load_stock_data(ticker, period="5y"):
    """
    Download stock data and prepare features.
    """

    # Download stock data
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False
    )

    # Fix Yahoo Finance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert index to normal column
    df.reset_index(inplace=True)

    # Previous day's close
    df["Previous Close"] = df["Close"].shift(1)

    # 5-Day Moving Average
    df["MA5"] = df["Close"].rolling(window=5).mean()

    # 10-Day Moving Average
    df["MA10"] = df["Close"].rolling(window=10).mean()

    # 20-Day Moving Average
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # Daily Percentage Return
    df["Daily Return"] = df["Close"].pct_change()

    # Daily Price Range
    df["Price Range"] = df["High"] - df["Low"]

    # Target = Next day's closing price
    df["Target"] = df["Close"].shift(-1)

    # Remove missing rows
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    df.index = df.index + 1
    
    return df
