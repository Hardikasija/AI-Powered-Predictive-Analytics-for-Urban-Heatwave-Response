import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover - optional for headless runs
    gpd = None
    Polygon = None


@dataclass
class SimulationConfig:
    start_date: str = "2020-06-01"
    days: int = 90
    freq: str = "H"
    n_stations: int = 20
    grid_size: int = 12
    city_center: Tuple[float, float] = (40.7128, -74.0060)
    seed: int = 42


def _daily_cycle(hour: int) -> float:
    return math.sin((2 * math.pi / 24) * (hour - 6))


def _seasonal_cycle(day_index: int, total_days: int) -> float:
    return math.sin((2 * math.pi / total_days) * day_index)


def generate_synthetic_dataset(output_dir: str, config: SimulationConfig) -> None:
    rng = np.random.default_rng(config.seed)
    os.makedirs(output_dir, exist_ok=True)

    timestamps = pd.date_range(config.start_date, periods=config.days * 24, freq=config.freq)

    # Stations
    lat_center, lon_center = config.city_center
    station_lats = lat_center + rng.normal(0, 0.05, size=config.n_stations)
    station_lons = lon_center + rng.normal(0, 0.05, size=config.n_stations)
    urban_index = rng.uniform(0.2, 1.0, size=config.n_stations)

    stations = pd.DataFrame(
        {
            "station_id": [f"S{i:02d}" for i in range(config.n_stations)],
            "lat": station_lats,
            "lon": station_lons,
            "urban_index": urban_index,
        }
    )

    # Climate timeseries
    records = []
    for station_id, u_index in zip(stations["station_id"], stations["urban_index"]):
        base_temp = 28 + 4 * u_index
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day = i // 24
            temp = (
                base_temp
                + 5 * _daily_cycle(hour)
                + 3 * _seasonal_cycle(day, config.days)
                + rng.normal(0, 1.5)
            )
            humidity = np.clip(70 - 0.5 * temp + rng.normal(0, 5), 20, 95)
            wind_speed = np.clip(rng.normal(3, 1), 0, 12)
            solar_rad = np.clip(600 * max(_daily_cycle(hour), 0) + rng.normal(0, 20), 0, 900)
            lst = temp + 2 * u_index + rng.normal(0, 0.8)
            ndvi = np.clip(0.6 - 0.4 * u_index + rng.normal(0, 0.05), 0.05, 0.8)
            ndbi = np.clip(0.2 + 0.6 * u_index + rng.normal(0, 0.05), 0.05, 0.95)
            pop_density = np.clip(5000 + 8000 * u_index + rng.normal(0, 500), 500, 20000)
            bld_density = np.clip(0.2 + 0.7 * u_index + rng.normal(0, 0.03), 0.05, 0.95)

            records.append(
                {
                    "station_id": station_id,
                    "timestamp": ts,
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "solar_radiation": solar_rad,
                    "lst": lst,
                    "ndvi": ndvi,
                    "ndbi": ndbi,
                    "population_density": pop_density,
                    "building_density": bld_density,
                }
            )

    climate = pd.DataFrame.from_records(records)

    # Urban grid indicators
    grid_cells = []
    grid_size = config.grid_size
    lat_min = lat_center - 0.1
    lat_max = lat_center + 0.1
    lon_min = lon_center - 0.1
    lon_max = lon_center + 0.1

    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)

    for i in range(grid_size):
        for j in range(grid_size):
            cell_id = f"G{i:02d}{j:02d}"
            cell_lat = (lat_edges[i] + lat_edges[i + 1]) / 2
            cell_lon = (lon_edges[j] + lon_edges[j + 1]) / 2
            urban_factor = np.clip(rng.normal(0.6, 0.2), 0.1, 1.0)
            ndvi = np.clip(0.65 - 0.45 * urban_factor + rng.normal(0, 0.05), 0.05, 0.85)
            ndbi = np.clip(0.25 + 0.55 * urban_factor + rng.normal(0, 0.05), 0.05, 0.95)
            impervious = np.clip(0.2 + 0.7 * urban_factor + rng.normal(0, 0.04), 0.05, 0.95)
            green_ratio = np.clip(0.7 - 0.6 * urban_factor + rng.normal(0, 0.05), 0.05, 0.9)
            pop_density = np.clip(4000 + 9000 * urban_factor + rng.normal(0, 600), 500, 22000)
            bld_density = np.clip(0.2 + 0.7 * urban_factor + rng.normal(0, 0.04), 0.05, 0.95)

            grid_cells.append(
                {
                    "grid_id": cell_id,
                    "lat": cell_lat,
                    "lon": cell_lon,
                    "ndvi": ndvi,
                    "ndbi": ndbi,
                    "impervious_ratio": impervious,
                    "green_coverage_ratio": green_ratio,
                    "population_density": pop_density,
                    "building_density": bld_density,
                }
            )

    grid_df = pd.DataFrame(grid_cells)

    # Save
    stations.to_csv(os.path.join(output_dir, "stations.csv"), index=False)
    climate.to_csv(os.path.join(output_dir, "climate_timeseries.csv"), index=False)
    grid_df.to_csv(os.path.join(output_dir, "urban_grid.csv"), index=False)

    if gpd is not None and Polygon is not None:
        polygons = []
        for i in range(grid_size):
            for j in range(grid_size):
                polygons.append(
                    Polygon(
                        [
                            (lon_edges[j], lat_edges[i]),
                            (lon_edges[j + 1], lat_edges[i]),
                            (lon_edges[j + 1], lat_edges[i + 1]),
                            (lon_edges[j], lat_edges[i + 1]),
                        ]
                    )
                )
        grid_gdf = gpd.GeoDataFrame(grid_df, geometry=polygons, crs="EPSG:4326")
        grid_gdf.to_file(os.path.join(output_dir, "urban_grid.geojson"), driver="GeoJSON")


def ensure_synthetic_data(base_dir: str, config: SimulationConfig) -> str:
    raw_dir = os.path.join(base_dir, "raw")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    climate_path = os.path.join(raw_dir, "climate_timeseries.csv")
    if not os.path.exists(climate_path):
        generate_synthetic_dataset(raw_dir, config)

    return raw_dir
