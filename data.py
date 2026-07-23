"""
data.py

Downloads historical stock data from Yahoo Finance
and prepares it for machine learning.
"""

import yfinance as yf
import pandas as pd


def load_stock_data(ticker, period="5y"):
    """
    Download stock data.

    Parameters
    ----------
    ticker : str
        Stock symbol.

    period : str
        Historical period.

    Returns
    -------
    DataFrame
    """

    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False
    )

    df.reset_index(inplace=True)

    # Previous day's close
    df["Previous Close"] = df["Close"].shift(1)

    # Target = Next day's closing price
    df["Target"] = df["Close"].shift(-1)

    # Remove missing rows
    df.dropna(inplace=True)

    return df