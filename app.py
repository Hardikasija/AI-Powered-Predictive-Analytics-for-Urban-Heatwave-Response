"""Streamlit app for urban heatwave risk prediction."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from prediction import load_metadata, predict_from_openweather, predict_from_user_inputs


st.set_page_config(page_title="Urban Heatwave Response", layout="wide")

st.title("AI-Powered Predictive Analytics for Urban Heatwave Response")
st.write(
    "Enter current weather and urban conditions to estimate heatwave risk for the next few days."
)

metadata_exists = Path("artifacts/metadata.json").exists()
if not metadata_exists:
    st.warning("Training artifacts are missing. Run `python main.py` before launching the app.")
    st.stop()

metadata = load_metadata()
available_models = []
for candidate in ["random_forest", "xgboost"]:
    model_path = Path("artifacts/models") / f"{candidate}_horizon_{metadata['forecast_horizon']}.joblib"
    if model_path.exists():
        available_models.append(candidate)

if not available_models:
    st.error("No tabular model artifacts were found. Run training again to create Random Forest or XGBoost models.")
    st.stop()

with st.sidebar:
    st.header("Model Settings")
    selected_model = st.selectbox("Model", options=available_models, index=0)
    forecast_horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=7, value=metadata["forecast_horizon"])
    st.caption(f"Alert threshold: {metadata['alert_threshold']:.2f}")

manual_tab, api_tab = st.tabs(["Manual Input", "OpenWeatherMap"])

with manual_tab:
    col1, col2, col3 = st.columns(3)

    with col1:
        temperature_max = st.number_input("Max temperature (C)", min_value=15.0, max_value=55.0, value=39.0, step=0.1)
        temperature_min = st.number_input("Min temperature (C)", min_value=5.0, max_value=40.0, value=28.0, step=0.1)
        temperature_avg = st.number_input("Avg temperature (C)", min_value=10.0, max_value=45.0, value=33.5, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=5.0, max_value=100.0, value=58.0, step=0.1)

    with col2:
        wind_speed = st.number_input("Wind speed (m/s)", min_value=0.0, max_value=20.0, value=2.2, step=0.1)
        air_pressure = st.number_input("Air pressure (hPa)", min_value=980.0, max_value=1040.0, value=1008.0, step=0.1)
        precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=300.0, value=1.0, step=0.1)
        solar_radiation = st.number_input("Solar radiation (W/m^2 proxy)", min_value=50.0, max_value=500.0, value=305.0, step=1.0)

    with col3:
        population_density = st.number_input("Population density", min_value=500, max_value=40000, value=12000, step=100)
        green_cover = st.number_input("Green cover ratio", min_value=0.0, max_value=1.0, value=0.16, step=0.01)
        built_up_index = st.number_input("Built-up index", min_value=0.0, max_value=1.0, value=0.74, step=0.01)

    if st.button("Predict Heatwave Risk", use_container_width=True):
        result = predict_from_user_inputs(
            inputs={
                "temperature_max": temperature_max,
                "temperature_min": temperature_min,
                "temperature_avg": temperature_avg,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "air_pressure": air_pressure,
                "precipitation": precipitation,
                "solar_radiation": solar_radiation,
                "population_density": population_density,
                "green_cover": green_cover,
                "built_up_index": built_up_index,
            },
            model_name=selected_model,
            forecast_horizon=forecast_horizon,
        )

        st.subheader("Prediction Result")
        st.metric("Heatwave probability", f"{result['heatwave_probability'] * 100:.2f}%")
        st.metric("Alert level", result["alert_level"])
        st.metric("Severity", result["severity"].title())
        st.write(f"Predicted heatwave: `{result['predicted_heatwave']}`")

        st.subheader("Recommended Actions")
        for action in result["recommended_actions"]:
            st.write(f"- {action}")

with api_tab:
    st.caption("Use an OpenWeatherMap API key to fetch live city forecasts and estimate heatwave risk.")
    api_key = st.text_input("OpenWeatherMap API key", type="password")
    city_name = st.text_input("City name", value="Delhi")
    api_population_density = st.number_input("Population density override", min_value=0, max_value=40000, value=0, step=100)
    api_green_cover = st.number_input("Green cover override", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    api_built_up_index = st.number_input("Built-up index override", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    if st.button("Predict From OpenWeatherMap", use_container_width=True):
        result = predict_from_openweather(
            city_name=city_name,
            api_key=api_key or None,
            model_name=selected_model,
            forecast_horizon=forecast_horizon,
            population_density=api_population_density or None,
            green_cover=api_green_cover or None,
            built_up_index=api_built_up_index or None,
        )

        st.subheader("Live Forecast Result")
        st.metric("Heatwave probability", f"{result['heatwave_probability'] * 100:.2f}%")
        st.metric("Alert level", result["alert_level"])
        st.metric("Severity", result["severity"].title())
        st.write(f"Predicted heatwave: `{result['predicted_heatwave']}`")
        st.caption(f"Source: {result['source']}")

        st.subheader("Recommended Actions")
        for action in result["recommended_actions"]:
            st.write(f"- {action}")
