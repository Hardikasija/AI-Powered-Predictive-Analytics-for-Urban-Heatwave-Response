# AI-Powered Predictive Analytics for Urban Heatwave Response

This project is a complete, beginner-friendly prototype for forecasting urban heatwave risk with machine learning. It now supports a real public daily climate dataset for training, OpenWeatherMap live forecast ingestion for operational scoring, compares multiple models, and serves actionable alerts through a Streamlit app.

## What the project does

- Downloads a real public multi-city daily climate dataset from NASA POWER
- Keeps a synthetic fallback dataset for offline experimentation
- Builds forecasting targets for heatwave events in the next `1-7` days
- Compares `Random Forest`, `XGBoost`, and `LSTM` models
- Handles missing values, scaling, feature engineering, and class imbalance
- Evaluates models with accuracy, precision, recall, F1-score, and ROC-AUC
- Produces temperature, ROC, and prediction-vs-actual plots
- Generates alert levels and heatwave severity classes
- Provides a Streamlit app for manual inputs and OpenWeatherMap live predictions

## Project structure

- `data_preprocessing.py` dataset simulation, cleaning, feature engineering, and target creation
- `data_ingestion.py` NASA POWER dataset download and OpenWeatherMap API ingestion
- `model_training.py` model training, cross-validation, evaluation, artifact saving, and plots
- `prediction.py` artifact loading, forecast generation, alert logic, and severity estimation
- `app.py` Streamlit web app
- `main.py` end-to-end pipeline runner
- `train_model.py` convenience wrapper for training
- `predict_heatwave.py` convenience wrapper for generating a sample forecast
- `artifacts/` saved models, scalers, metadata, and reports
- `outputs/` generated plots and prediction summaries

## Features used

- Weather:
  - `temperature_max`
  - `temperature_min`
  - `temperature_avg`
  - `humidity`
  - `wind_speed`
  - `air_pressure`
  - `precipitation`
  - `solar_radiation`
- Urban:
  - `population_density`
  - `green_cover`
  - `built_up_index`

## Heatwave definition

The synthetic pipeline marks a day as a heatwave when high temperature, elevated average temperature, strong solar load, and low rainfall align with urban heat island effects. Forecast targets are created as:

- `heatwave_next_1d`
- `heatwave_next_2d`
- ...
- `heatwave_next_7d`

Each target answers: "Will a heatwave occur within the next N days?"

## Installation

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Data sources

- Training dataset:
  - NASA POWER Daily API public climate data
  - Variables used: `T2M_MAX`, `T2M_MIN`, `T2M`, `RH2M`, `WS2M`, `PS`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`
- Live inference:
  - OpenWeatherMap Geocoding API
  - OpenWeatherMap One Call API 3.0

## Run the full project

```bash
.\.venv\Scripts\python main.py
```

By default, that command will:

1. Fetch a real public multi-city daily climate dataset
2. Preprocess and engineer features
3. Train and evaluate models for a default `3-day` forecast horizon
4. Save artifacts to `artifacts/`
5. Save plots and summaries to `outputs/`

To use the simulation fallback instead:

```bash
.\.venv\Scripts\python main.py --data-source simulation --regenerate-data
```

## Train a specific horizon

```bash
.\.venv\Scripts\python train_model.py --horizon 5
```

## Generate a sample forecast

```bash
.\.venv\Scripts\python predict_heatwave.py --horizon 3 --model random_forest
```

## Launch the web app

```bash
.\.venv\Scripts\python -m streamlit run app.py
```

## Use OpenWeatherMap live scoring

Set your API key in an environment variable:

```bash
$env:OPENWEATHERMAP_API_KEY="your_api_key_here"
```

OpenWeatherMap One Call 3.0 requires a valid API key and account access for that product. Then use the Streamlit app or extend the prediction utilities in `prediction.py`.

## Output files

- `artifacts/models/` trained model files
- `artifacts/reports/metrics_horizon_*.json` evaluation report
- `artifacts/reports/comparison_horizon_*.csv` model comparison table
- `artifacts/reports/forecast_sample_horizon_*.json` sample alert output
- `outputs/temperature_trends.png`
- `outputs/roc_curve_horizon_*.png`
- `outputs/predictions_vs_actual_horizon_*.png`

## Notes

- `XGBoost` and `TensorFlow` are used when available. If either package is missing, the pipeline skips that model gracefully and records the reason in the metrics report.
- The LSTM uses sequence windows built from daily features for time-series classification.
- The project uses a time-based `80/20` split and `TimeSeriesSplit` cross-validation on the training segment.
- OpenWeatherMap daily forecast responses do not expose direct solar radiation. For live inference, the app uses a solar-load proxy derived from UV index.
