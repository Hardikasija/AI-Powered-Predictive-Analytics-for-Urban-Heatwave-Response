"""Convenience training entry point."""

from __future__ import annotations

import argparse

from data_preprocessing import SimulationConfig, simulate_dataset
from data_ingestion import build_public_climate_dataset
from model_training import TrainingConfig, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train urban heatwave forecasting models")
    parser.add_argument("--horizon", type=int, default=3, choices=range(1, 8), help="Forecast horizon in days")
    parser.add_argument(
        "--data-source",
        type=str,
        default="real",
        choices=["real", "simulation"],
        help="Choose the data source used for training",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate the selected dataset before training",
    )
    args = parser.parse_args()

    if args.regenerate_data and args.data_source == "simulation":
        simulate_dataset("data/urban_heatwave_simulated_daily.csv", SimulationConfig())
    if args.regenerate_data and args.data_source == "real":
        build_public_climate_dataset("data/urban_heatwave_real_daily.csv")

    train_and_evaluate(TrainingConfig(forecast_horizon=args.horizon, data_source=args.data_source))
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
