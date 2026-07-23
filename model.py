"""
model.py

Contains functions to train and evaluate
Random Forest Regression model.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)


def train_model(df):

    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Previous Close"
    ]

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    return (
        model,
        X_test,
        y_test,
        predictions,
        r2,
        mae,
        rmse
    )