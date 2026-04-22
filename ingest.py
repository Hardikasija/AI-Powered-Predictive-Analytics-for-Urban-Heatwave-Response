import os
import pandas as pd


def load_raw_data(raw_dir: str) -> dict:
    climate = pd.read_csv(os.path.join(raw_dir, "climate_timeseries.csv"), parse_dates=["timestamp"])
    stations = pd.read_csv(os.path.join(raw_dir, "stations.csv"))
    grid = pd.read_csv(os.path.join(raw_dir, "urban_grid.csv"))
    return {"climate": climate, "stations": stations, "grid": grid}
