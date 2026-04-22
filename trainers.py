import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from .evaluate import regression_metrics, classification_metrics

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:  # pragma: no cover
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    lstm_lookback: int = 24


def train_regression_models(
    df: pd.DataFrame, feature_cols: list, target_col: str, output_dir: str, config: TrainConfig
) -> Tuple[Dict[str, dict], Dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    results = {}
    models = {}

    # Linear Regression
    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_test)
    results["linear_regression"] = regression_metrics(y_test, y_pred)
    models["linear_regression"] = lin
    joblib.dump(lin, os.path.join(output_dir, "linear_regression.joblib"))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=config.random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["random_forest"] = regression_metrics(y_test, y_pred)
    models["random_forest"] = rf
    joblib.dump(rf, os.path.join(output_dir, "random_forest.joblib"))

    # XGBoost
    if xgb is not None:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        results["xgboost"] = regression_metrics(y_test, y_pred)
        models["xgboost"] = xgb_model
        joblib.dump(xgb_model, os.path.join(output_dir, "xgboost.joblib"))

    return results, models


def train_heatwave_classifier(
    df: pd.DataFrame, feature_cols: list, target_col: str, output_dir: str, config: TrainConfig
) -> Dict[str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    rf_clf = RandomForestClassifier(n_estimators=200, random_state=config.random_state)
    rf_clf.fit(X_train, y_train)
    y_pred = (rf_clf.predict(X_test) > 0.5).astype(int)

    metrics = classification_metrics(y_test, y_pred)
    joblib.dump(rf_clf, os.path.join(output_dir, "heatwave_classifier.joblib"))

    return {"heatwave_classifier": metrics}


def _make_lstm_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i : i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


def train_lstm_model(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    config: TrainConfig,
) -> dict:
    if tf is None:
        return {"lstm": "TensorFlow not installed"}

    os.makedirs(output_dir, exist_ok=True)
    series = df.sort_values("timestamp")[target_col].values.astype(np.float32)
    X, y = _make_lstm_sequences(series, config.lstm_lookback)

    split = int(len(X) * (1 - config.test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential(
        [
            LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train[..., None], y_train, epochs=10, batch_size=64, verbose=0)

    y_pred = model.predict(X_test[..., None], verbose=0).squeeze()
    metrics = regression_metrics(y_test, y_pred)

    model.save(os.path.join(output_dir, "lstm_model"))
    return {"lstm": metrics}
