import numpy as np
import pandas as pd


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Urban Heat Island intensity: LST - temp (proxy for urban-rural contrast)
    df["uhi_intensity"] = df["lst"] - df["temperature"]

    # Impervious surface ratio from NDBI proxy
    df["impervious_ratio"] = np.clip(0.1 + 0.9 * df["ndbi"], 0, 1)

    # Green coverage ratio from NDVI proxy
    df["green_coverage_ratio"] = np.clip(df["ndvi"], 0, 1)

    # Urban density indicator
    df["urban_density_index"] = (
        0.5 * (df["population_density"] / df["population_density"].max())
        + 0.5 * df["building_density"]
    )

    return df
