"""Data simulation and preprocessing utilities for urban heatwave forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_ingestion import build_public_climate_dataset


@dataclass
class SimulationConfig:
    """Configuration for the synthetic daily urban dataset."""

    start_date: str = "2018-01-01"
    years: int = 6
    n_zones: int = 8
    seed: int = 42
    missing_rate: float = 0.03


def _seasonal_signal(day_of_year: np.ndarray) -> np.ndarray:
    return np.sin((2 * np.pi * day_of_year) / 365.25)


def _monsoon_signal(day_of_year: np.ndarray) -> np.ndarray:
    # Gentle rainy-season bump to make precipitation seasonally realistic.
    return np.exp(-((day_of_year - 220) ** 2) / (2 * 35**2))


def simulate_dataset(
    output_path: str | Path = "data/urban_heatwave_daily.csv",
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Generate a multi-zone urban daily dataset and save it to disk."""

    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    periods = int(round(config.years * 365.25))
    dates = pd.date_range(config.start_date, periods=periods, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    seasonal = _seasonal_signal(day_of_year)
    monsoon = _monsoon_signal(day_of_year)

    records: List[Dict[str, float]] = []
    for zone_idx in range(config.n_zones):
        zone_name = f"Zone_{zone_idx + 1}"
        urban_intensity = rng.uniform(0.25, 0.95)
        population_density = rng.integers(4500, 18000)
        green_cover = np.clip(rng.uniform(0.08, 0.45) - 0.10 * urban_intensity, 0.03, 0.55)
        built_up_index = np.clip(0.25 + 0.70 * urban_intensity + rng.normal(0, 0.04), 0.2, 0.98)

        zone_offset = rng.normal(0, 0.9)
        short_heatwave_boost = np.zeros(periods)

        # Add a few persistent hot spells each year.
        for year_start in range(0, periods, 365):
            year_end = min(year_start + 365, periods)
            latest_valid_start = year_end - 8
            if latest_valid_start <= year_start + 90:
                continue

            for _ in range(rng.integers(2, 5)):
                start = int(rng.integers(year_start + 90, min(year_start + 280, latest_valid_start)))
                duration = int(rng.integers(3, 8))
                intensity = rng.uniform(2.5, 5.5)
                short_heatwave_boost[start : start + duration] += intensity

        avg_temp = (
            28
            + 8.0 * seasonal
            + 2.2 * urban_intensity
            + zone_offset
            + short_heatwave_boost
            + rng.normal(0, 1.1, periods)
        )
        temp_range = np.clip(7 + 1.3 * (1 - monsoon) + rng.normal(0, 0.8, periods), 4, 13)
        temperature_max = avg_temp + temp_range / 2 + rng.normal(0, 0.6, periods)
        temperature_min = avg_temp - temp_range / 2 + rng.normal(0, 0.6, periods)
        humidity = np.clip(
            66 - 0.7 * (avg_temp - 30) + 16 * monsoon - 8 * urban_intensity + rng.normal(0, 5, periods),
            20,
            98,
        )
        wind_speed = np.clip(2.8 + 1.0 * monsoon - 0.5 * urban_intensity + rng.normal(0, 0.7, periods), 0.2, 10)
        air_pressure = 1012 - 0.18 * (avg_temp - 30) + rng.normal(0, 2.0, periods)
        precipitation = np.clip(2 + 14 * monsoon + rng.gamma(1.5, 1.3, periods) - 1.8 * short_heatwave_boost, 0, None)
        solar_radiation = np.clip(
            230 + 110 * seasonal + 55 * (1 - monsoon) + 25 * urban_intensity + rng.normal(0, 18, periods),
            80,
            420,
        )

        heat_index_proxy = (
            0.55 * temperature_max
            + 0.20 * avg_temp
            + 0.10 * solar_radiation / 10
            + 0.05 * humidity / 5
            + 0.10 * built_up_index * 10
            - 0.12 * wind_speed
            - 0.03 * precipitation
        )

        daily_heatwave = (
            (temperature_max >= 40)
            & (avg_temp >= 33)
            & (solar_radiation >= 250)
            & (heat_index_proxy >= 30)
        ).astype(int)

        severity = np.where(
            daily_heatwave == 0,
            "none",
            np.where(
                heat_index_proxy < 31.5,
                "mild",
                np.where(heat_index_proxy < 33.5, "moderate", "extreme"),
            ),
        )

        for idx, current_date in enumerate(dates):
            records.append(
                {
                    "date": current_date,
                    "zone": zone_name,
                    "temperature_max": round(float(temperature_max[idx]), 2),
                    "temperature_min": round(float(temperature_min[idx]), 2),
                    "temperature_avg": round(float(avg_temp[idx]), 2),
                    "humidity": round(float(humidity[idx]), 2),
                    "wind_speed": round(float(wind_speed[idx]), 2),
                    "air_pressure": round(float(air_pressure[idx]), 2),
                    "precipitation": round(float(precipitation[idx]), 2),
                    "solar_radiation": round(float(solar_radiation[idx]), 2),
                    "population_density": int(population_density),
                    "green_cover": round(float(green_cover), 3),
                    "built_up_index": round(float(built_up_index), 3),
                    "urban_heatwave_event": int(daily_heatwave[idx]),
                    "heatwave_severity": severity[idx],
                }
            )

    df = pd.DataFrame(records)
    df = inject_missing_values(df, missing_rate=config.missing_rate, seed=config.seed)
    df.to_csv(output_path, index=False)
    return df


def inject_missing_values(df: pd.DataFrame, missing_rate: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """Inject light missingness to exercise imputation logic."""

    rng = np.random.default_rng(seed)
    result = df.copy()
    numeric_columns = [
        column
        for column in result.select_dtypes(include=["number"]).columns
        if column not in {"urban_heatwave_event", "population_density"}
    ]

    for column in numeric_columns:
        mask = rng.random(len(result)) < missing_rate
        result.loc[mask, column] = np.nan

    return result


def _label_heatwaves(df: pd.DataFrame) -> pd.DataFrame:
    """Generate event and severity labels from real or simulated climate signals."""

    labeled = df.copy()
    zone_group = labeled.groupby("zone")

    temp_max_threshold = zone_group["temperature_max"].transform(lambda values: values.quantile(0.90))
    temp_avg_threshold = zone_group["temperature_avg"].transform(lambda values: values.quantile(0.85))
    solar_threshold = zone_group["solar_radiation"].transform(lambda values: values.quantile(0.65))
    precipitation_threshold = zone_group["precipitation"].transform(lambda values: values.quantile(0.60))

    heat_score = (
        (labeled["temperature_max"] - temp_max_threshold).clip(lower=0) * 0.45
        + (labeled["temperature_avg"] - temp_avg_threshold).clip(lower=0) * 0.35
        + (labeled["solar_radiation"] - solar_threshold).clip(lower=0) * 0.02
        + (precipitation_threshold - labeled["precipitation"]).clip(lower=0) * 0.08
        + labeled["built_up_index"] * 0.8
        - labeled["green_cover"] * 0.7
    )

    event = (
        (labeled["temperature_max"] >= temp_max_threshold)
        & (labeled["temperature_avg"] >= temp_avg_threshold)
        & (labeled["solar_radiation"] >= solar_threshold)
        & (labeled["precipitation"] <= precipitation_threshold)
    ).astype(int)

    severity = np.where(
        event == 0,
        "none",
        np.where(heat_score < 1.2, "mild", np.where(heat_score < 2.3, "moderate", "extreme")),
    )

    labeled["urban_heatwave_event"] = event
    labeled["heatwave_severity"] = severity
    return labeled


def load_or_create_dataset(
    dataset_path: str | Path = "data/urban_heatwave_real_daily.csv",
    data_source: str = "real",
    allow_fallback_to_simulation: bool = False,
) -> pd.DataFrame:
    """Load an existing dataset or fetch/create one based on the requested source."""

    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        df = pd.read_csv(dataset_path, parse_dates=["date"])
    else:
        if data_source == "real":
            try:
                df = build_public_climate_dataset(dataset_path)
            except Exception:
                if not allow_fallback_to_simulation:
                    raise
                df = simulate_dataset(dataset_path.with_name("urban_heatwave_simulated_daily.csv"))
        else:
            df = simulate_dataset(dataset_path)

    df["date"] = pd.to_datetime(df["date"])
    if "urban_heatwave_event" not in df.columns or "heatwave_severity" not in df.columns:
        df = _label_heatwaves(df)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged, rolling, seasonal, and urban heat features."""

    work_df = df.sort_values(["zone", "date"]).reset_index(drop=True).copy()
    work_df["day_of_year"] = work_df["date"].dt.dayofyear
    work_df["month"] = work_df["date"].dt.month
    work_df["week_of_year"] = work_df["date"].dt.isocalendar().week.astype(int)
    work_df["day_sin"] = np.sin(2 * np.pi * work_df["day_of_year"] / 365.25)
    work_df["day_cos"] = np.cos(2 * np.pi * work_df["day_of_year"] / 365.25)
    work_df["temp_range"] = work_df["temperature_max"] - work_df["temperature_min"]
    work_df["dryness_index"] = work_df["solar_radiation"] / (work_df["humidity"] + 1)
    work_df["comfort_index"] = work_df["temperature_avg"] - 0.55 * (1 - work_df["humidity"] / 100) * (
        work_df["temperature_avg"] - 14.5
    )
    work_df["urban_heat_risk_index"] = (
        0.35 * work_df["temperature_max"]
        + 0.20 * work_df["temperature_avg"]
        + 0.15 * work_df["solar_radiation"] / 10
        + 0.10 * work_df["population_density"] / 1000
        + 0.10 * work_df["built_up_index"] * 10
        - 0.12 * work_df["green_cover"] * 10
        - 0.08 * work_df["wind_speed"]
        - 0.03 * work_df["precipitation"]
    )

    group = work_df.groupby("zone", group_keys=False)
    base_columns = [
        "temperature_max",
        "temperature_min",
        "temperature_avg",
        "humidity",
        "wind_speed",
        "air_pressure",
        "precipitation",
        "solar_radiation",
        "urban_heat_risk_index",
    ]

    for column in base_columns:
        work_df[f"{column}_lag_1"] = group[column].shift(1)
        work_df[f"{column}_lag_3"] = group[column].shift(3)
        work_df[f"{column}_rolling_mean_3"] = group[column].transform(lambda values: values.rolling(3).mean())
        work_df[f"{column}_rolling_mean_7"] = group[column].transform(lambda values: values.rolling(7).mean())
        work_df[f"{column}_rolling_std_7"] = group[column].transform(lambda values: values.rolling(7).std())

    return work_df


def create_forecast_targets(df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """Create binary forecast targets for heatwaves occurring in the next N days."""

    target_col = f"heatwave_next_{forecast_horizon}d"
    work_df = df.sort_values(["zone", "date"]).reset_index(drop=True).copy()

    future_windows = []
    group = work_df.groupby("zone")["urban_heatwave_event"]
    for step in range(1, forecast_horizon + 1):
        future_windows.append(group.shift(-step).fillna(0))

    stacked = np.column_stack(future_windows)
    work_df[target_col] = (stacked.max(axis=1) > 0).astype(int)

    severity_rank = {"none": 0, "mild": 1, "moderate": 2, "extreme": 3}
    severity_numeric = work_df["heatwave_severity"].map(severity_rank)
    severity_future = []
    severity_group = severity_numeric.groupby(work_df["zone"])
    for step in range(1, forecast_horizon + 1):
        severity_future.append(severity_group.shift(-step).fillna(0))
    severity_stack = np.column_stack(severity_future)
    work_df[f"severity_next_{forecast_horizon}d"] = severity_stack.max(axis=1).astype(int)

    return work_df


def prepare_dataset(
    dataset_path: str | Path | None = None,
    forecast_horizon: int = 3,
    data_source: str = "real",
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load, clean, engineer, and label the dataset for model training."""

    if dataset_path is None:
        dataset_path = (
            "data/urban_heatwave_real_daily.csv"
            if data_source == "real"
            else "data/urban_heatwave_simulated_daily.csv"
        )
    df = load_or_create_dataset(dataset_path, data_source=data_source)
    df = engineer_features(df)
    df = create_forecast_targets(df, forecast_horizon=forecast_horizon)

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    group = df.groupby("zone", group_keys=False)
    df[numeric_columns] = group[numeric_columns].apply(lambda block: block.interpolate(limit_direction="both"))
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median(numeric_only=True))

    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=int)
    df = pd.concat([df, zone_dummies], axis=1)

    feature_columns = [
        "temperature_max",
        "temperature_min",
        "temperature_avg",
        "humidity",
        "wind_speed",
        "air_pressure",
        "precipitation",
        "solar_radiation",
        "population_density",
        "green_cover",
        "built_up_index",
        "day_of_year",
        "month",
        "week_of_year",
        "day_sin",
        "day_cos",
        "temp_range",
        "dryness_index",
        "comfort_index",
        "urban_heat_risk_index",
    ]

    lag_roll_columns = [
        column
        for column in df.columns
        if any(tag in column for tag in ["_lag_", "_rolling_mean_", "_rolling_std_"])
    ]
    feature_columns.extend(sorted(lag_roll_columns))
    feature_columns.extend(zone_dummies.columns.tolist())

    target_column = f"heatwave_next_{forecast_horizon}d"
    required_columns = feature_columns + [target_column, f"severity_next_{forecast_horizon}d"]
    prepared = df.dropna(subset=required_columns).reset_index(drop=True)

    return prepared, feature_columns, target_column


def time_based_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time to preserve chronological order."""

    ordered = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    train_df = ordered.iloc[:split_index].copy()
    test_df = ordered.iloc[split_index:].copy()

    # Make sure the train set includes both classes when possible.
    if train_df[target_column].nunique() < 2 or test_df[target_column].nunique() < 2:
        ordered = df.sort_values(["zone", "date"]).reset_index(drop=True)
        split_index = int(len(ordered) * (1 - test_size))
        train_df = ordered.iloc[:split_index].copy()
        test_df = ordered.iloc[split_index:].copy()

    return train_df, test_df
