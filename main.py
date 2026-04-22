"""End-to-end runner for AI-powered urban heatwave prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preprocessing import SimulationConfig, simulate_dataset
from data_ingestion import build_public_climate_dataset
from model_training import TrainingConfig, train_and_evaluate
from prediction import predict_next_heatwave


def run_pipeline(horizon: int = 3, force_regenerate: bool = False, data_source: str = "real") -> None:
    dataset_path = Path("data/urban_heatwave_real_daily.csv" if data_source == "real" else "data/urban_heatwave_simulated_daily.csv")
    if data_source == "simulation" and (force_regenerate or not dataset_path.exists()):
        simulate_dataset(dataset_path, SimulationConfig())
    if data_source == "real" and (force_regenerate or not dataset_path.exists()):
        build_public_climate_dataset(dataset_path)

    results = train_and_evaluate(TrainingConfig(forecast_horizon=horizon, data_source=data_source))
    forecast = predict_next_heatwave(forecast_horizon=horizon)

    forecast_path = Path("artifacts/reports") / f"forecast_sample_horizon_{horizon}.json"
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_path.write_text(json.dumps(forecast, indent=2), encoding="utf-8")

    print("Urban heatwave forecasting pipeline completed.")
    print(f"Forecast horizon: {horizon} day(s)")
    print(f"Best model: {results['metadata']['best_model']}")
    print(f"Saved forecast sample to: {forecast_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Predictive Analytics for Urban Heatwave Response")
    parser.add_argument("--horizon", type=int, default=3, choices=range(1, 8), help="Forecast horizon in days")
    parser.add_argument(
        "--data-source",
        type=str,
        default="real",
        choices=["real", "simulation"],
        help="Choose between a real public dataset and the synthetic fallback dataset",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate the selected dataset even if a saved CSV already exists",
    )
    args = parser.parse_args()
    run_pipeline(horizon=args.horizon, force_regenerate=args.regenerate_data, data_source=args.data_source)
