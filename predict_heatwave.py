"""Convenience prediction entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction import predict_next_heatwave


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample urban heatwave forecast")
    parser.add_argument("--horizon", type=int, default=3, choices=range(1, 8), help="Forecast horizon in days")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["random_forest", "xgboost", "lstm"],
        help="Model to use for prediction",
    )
    args = parser.parse_args()

    prediction = predict_next_heatwave(model_name=args.model, forecast_horizon=args.horizon)
    output_path = Path("artifacts/reports") / f"manual_prediction_horizon_{args.horizon}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    print(json.dumps(prediction, indent=2))
    print(f"Saved prediction to: {output_path}")


if __name__ == "__main__":
    main()
