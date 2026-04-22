import os
import pandas as pd


def preprocess_data(climate: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = climate.merge(stations, on="station_id", how="left")

    # Basic cleaning
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
    df = df.dropna().reset_index(drop=True)

    return df


def save_processed(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "processed_timeseries.csv")
    df.to_csv(path, index=False)
    return path
