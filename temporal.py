import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["season_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)

    # Lag features
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    for lag in [1, 24, 48]:
        df[f"temp_lag_{lag}"] = df.groupby("station_id")["temperature"].shift(lag)

    # Rolling stats
    df["temp_roll_24"] = (
        df.groupby("station_id")["temperature"].rolling(24).mean().reset_index(level=0, drop=True)
    )
    df["temp_roll_72"] = (
        df.groupby("station_id")["temperature"].rolling(72).mean().reset_index(level=0, drop=True)
    )

    # Heat accumulation index
    df["heat_accum_index"] = (
        df.groupby("station_id")["temperature"].rolling(48).apply(lambda x: np.sum(np.maximum(x - 30, 0)), raw=True)
        .reset_index(level=0, drop=True)
    )

    return df
