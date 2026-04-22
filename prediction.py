"""Prediction and alerting utilities for urban heatwave forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import prepare_dataset
from data_ingestion import build_openweather_feature_payload

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def load_metadata(path: str | Path = "artifacts/metadata.json") -> Dict:
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError("Metadata not found. Run training first with `python main.py`.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _severity_from_probability(probability: float) -> str:
    if probability < 0.35:
        return "none"
    if probability < 0.60:
        return "mild"
    if probability < 0.80:
        return "moderate"
    return "extreme"


def _alert_level(probability: float, threshold: float) -> str:
    if probability < threshold * 0.75:
        return "Monitor"
    if probability < threshold:
        return "Watch"
    if probability < min(threshold + 0.20, 0.90):
        return "Warning"
    return "Emergency"


def _recommendations(alert_level: str) -> list[str]:
    guidance = {
        "Monitor": [
            "Continue routine monitoring of weather and environmental indicators.",
            "Review cooling center readiness and public communication channels.",
        ],
        "Watch": [
            "Prepare health services and emergency management teams for potential heat stress cases.",
            "Issue advisories for vulnerable populations such as elderly residents and outdoor workers.",
        ],
        "Warning": [
            "Activate local heat action plans and extend cooling center operating hours.",
            "Coordinate water distribution, shaded public spaces, and rapid outreach in high-risk zones.",
        ],
        "Emergency": [
            "Trigger emergency heatwave response protocols immediately.",
            "Deploy community response teams, medical support, and targeted public alerts without delay.",
        ],
    }
    return guidance[alert_level]


def _prepare_latest_sample(
    forecast_horizon: int,
    feature_columns: list[str],
    data_source: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    prepared_df, _, _ = prepare_dataset(forecast_horizon=forecast_horizon, data_source=data_source)
    latest = prepared_df.sort_values(["date", "zone"]).groupby("zone").tail(1).copy()
    sample_row = latest.iloc[[0]][feature_columns].copy()
    sample_context = latest.iloc[0]
    return sample_row, sample_context


def _predict_with_lstm(
    forecast_horizon: int,
    feature_columns: list[str],
    threshold: float,
) -> Dict[str, object]:
    if tf is None:
        raise RuntimeError("TensorFlow is not installed, so the LSTM model is unavailable.")

    model_path = Path("artifacts/models") / f"lstm_horizon_{forecast_horizon}.keras"
    preprocessor_path = Path("artifacts/models") / f"lstm_preprocessor_horizon_{forecast_horizon}.joblib"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("LSTM artifacts not found. Train the LSTM model first.")

    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    prepared_df, _, target_column = prepare_dataset(
        forecast_horizon=forecast_horizon,
        data_source=load_metadata()["data_source"],
    )
    sequence_length = load_metadata()["lstm_sequence_length"]
    latest_zone = prepared_df["zone"].iloc[0]
    zone_df = prepared_df[prepared_df["zone"] == latest_zone].sort_values("date").tail(sequence_length).copy()

    if len(zone_df) < sequence_length:
        raise ValueError("Not enough rows are available to build an LSTM prediction sequence.")

    transformed = preprocessor.transform(zone_df[feature_columns])
    features = transformed.reshape(1, sequence_length, len(feature_columns))
    probability = float(model.predict(features, verbose=0)[0][0])
    severity = _severity_from_probability(probability)
    alert_level = _alert_level(probability, threshold)

    return {
        "model": "lstm",
        "zone": latest_zone,
        "date": str(zone_df["date"].iloc[-1].date()),
        "heatwave_probability": round(probability, 4),
        "predicted_heatwave": bool(probability >= threshold),
        "alert_level": alert_level,
        "severity": severity,
        "recommended_actions": _recommendations(alert_level),
        "target_column": target_column,
    }


def predict_next_heatwave(
    model_name: str | None = None,
    forecast_horizon: int | None = None,
    threshold: float | None = None,
) -> Dict[str, object]:
    """Load saved artifacts and predict heatwave risk for a recent sample."""

    metadata = load_metadata()
    forecast_horizon = forecast_horizon or metadata["forecast_horizon"]
    model_name = model_name or metadata["best_model"]
    threshold = threshold or metadata["alert_threshold"]
    feature_columns = metadata["feature_columns"]
    data_source = metadata.get("data_source", "real")

    if model_name == "lstm":
        return _predict_with_lstm(forecast_horizon, feature_columns, threshold)

    sample_row, sample_context = _prepare_latest_sample(forecast_horizon, feature_columns, data_source)
    model_path = Path("artifacts/models") / f"{model_name}_horizon_{forecast_horizon}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")

    model = joblib.load(model_path)
    probability = float(model.predict_proba(sample_row)[:, 1][0])
    predicted_label = bool(probability >= threshold)
    severity = _severity_from_probability(probability)
    alert_level = _alert_level(probability, threshold)

    return {
        "model": model_name,
        "zone": sample_context["zone"],
        "date": str(pd.to_datetime(sample_context["date"]).date()),
        "heatwave_probability": round(probability, 4),
        "predicted_heatwave": predicted_label,
        "alert_level": alert_level,
        "severity": severity,
        "recommended_actions": _recommendations(alert_level),
        "temperature_max": float(sample_context["temperature_max"]),
        "temperature_avg": float(sample_context["temperature_avg"]),
        "humidity": float(sample_context["humidity"]),
        "forecast_horizon_days": forecast_horizon,
    }


def predict_from_user_inputs(
    inputs: Dict[str, float],
    model_name: str | None = None,
    forecast_horizon: int | None = None,
    threshold: float | None = None,
) -> Dict[str, object]:
    """Predict heatwave risk from app-provided user features."""

    metadata = load_metadata()
    forecast_horizon = forecast_horizon or metadata["forecast_horizon"]
    model_name = model_name or metadata["best_model"]
    threshold = threshold or metadata["alert_threshold"]

    if model_name == "lstm":
        raise ValueError("The web app uses tabular models for manual input. Choose Random Forest or XGBoost.")

    feature_columns = metadata["feature_columns"]
    model_path = Path("artifacts/models") / f"{model_name}_horizon_{forecast_horizon}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")

    base_frame, _, _ = prepare_dataset(
        forecast_horizon=forecast_horizon,
        data_source=metadata.get("data_source", "real"),
    )
    reference_row = base_frame.sort_values("date").iloc[-1:].copy()

    for key, value in inputs.items():
        if key in reference_row.columns:
            reference_row.loc[:, key] = value

    # Refresh a few derived features from current weather conditions.
    reference_row.loc[:, "temp_range"] = reference_row["temperature_max"] - reference_row["temperature_min"]
    reference_row.loc[:, "dryness_index"] = reference_row["solar_radiation"] / (reference_row["humidity"] + 1)
    reference_row.loc[:, "comfort_index"] = reference_row["temperature_avg"] - 0.55 * (
        1 - reference_row["humidity"] / 100
    ) * (reference_row["temperature_avg"] - 14.5)
    reference_row.loc[:, "urban_heat_risk_index"] = (
        0.35 * reference_row["temperature_max"]
        + 0.20 * reference_row["temperature_avg"]
        + 0.15 * reference_row["solar_radiation"] / 10
        + 0.10 * reference_row["population_density"] / 1000
        + 0.10 * reference_row["built_up_index"] * 10
        - 0.12 * reference_row["green_cover"] * 10
        - 0.08 * reference_row["wind_speed"]
        - 0.03 * reference_row["precipitation"]
    )

    model = joblib.load(model_path)
    probability = float(model.predict_proba(reference_row[feature_columns])[:, 1][0])
    alert_level = _alert_level(probability, threshold)
    severity = _severity_from_probability(probability)

    return {
        "model": model_name,
        "forecast_horizon_days": forecast_horizon,
        "heatwave_probability": round(probability, 4),
        "predicted_heatwave": bool(probability >= threshold),
        "alert_level": alert_level,
        "severity": severity,
        "recommended_actions": _recommendations(alert_level),
    }


def predict_from_openweather(
    city_name: str,
    api_key: str | None = None,
    model_name: str | None = None,
    forecast_horizon: int | None = None,
    threshold: float | None = None,
    population_density: int | None = None,
    green_cover: float | None = None,
    built_up_index: float | None = None,
) -> Dict[str, object]:
    """Generate a heatwave prediction directly from OpenWeatherMap forecast data."""

    metadata = load_metadata()
    forecast_horizon = forecast_horizon or metadata["forecast_horizon"]
    model_name = model_name or metadata["best_model"]
    threshold = threshold or metadata["alert_threshold"]

    if model_name == "lstm":
        raise ValueError("OpenWeatherMap live scoring currently supports Random Forest and XGBoost models.")

    feature_columns = metadata["feature_columns"]
    model_path = Path("artifacts/models") / f"{model_name}_horizon_{forecast_horizon}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")

    payload = build_openweather_feature_payload(
        city_name=city_name,
        api_key=api_key,
        horizon_days=forecast_horizon,
        population_density=population_density,
        green_cover=green_cover,
        built_up_index=built_up_index,
    )

    base_frame, _, _ = prepare_dataset(
        forecast_horizon=forecast_horizon,
        data_source=metadata.get("data_source", "real"),
    )
    reference_row = base_frame.sort_values("date").iloc[-1:].copy()

    for key, value in payload.items():
        if key in reference_row.columns:
            reference_row.loc[:, key] = value

    reference_row.loc[:, "temp_range"] = reference_row["temperature_max"] - reference_row["temperature_min"]
    reference_row.loc[:, "dryness_index"] = reference_row["solar_radiation"] / (reference_row["humidity"] + 1)
    reference_row.loc[:, "comfort_index"] = reference_row["temperature_avg"] - 0.55 * (
        1 - reference_row["humidity"] / 100
    ) * (reference_row["temperature_avg"] - 14.5)
    reference_row.loc[:, "urban_heat_risk_index"] = (
        0.35 * reference_row["temperature_max"]
        + 0.20 * reference_row["temperature_avg"]
        + 0.15 * reference_row["solar_radiation"] / 10
        + 0.10 * reference_row["population_density"] / 1000
        + 0.10 * reference_row["built_up_index"] * 10
        - 0.12 * reference_row["green_cover"] * 10
        - 0.08 * reference_row["wind_speed"]
        - 0.03 * reference_row["precipitation"]
    )

    model = joblib.load(model_path)
    probability = float(model.predict_proba(reference_row[feature_columns])[:, 1][0])
    alert_level = _alert_level(probability, threshold)
    severity = _severity_from_probability(probability)

    return {
        "city": payload["zone"],
        "model": model_name,
        "forecast_horizon_days": forecast_horizon,
        "heatwave_probability": round(probability, 4),
        "predicted_heatwave": bool(probability >= threshold),
        "alert_level": alert_level,
        "severity": severity,
        "recommended_actions": _recommendations(alert_level),
        "source": "OpenWeatherMap One Call 3.0",
    }
