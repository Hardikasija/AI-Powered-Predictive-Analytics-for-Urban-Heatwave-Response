"""External data ingestion for public climate data and OpenWeatherMap forecasts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests


NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OPENWEATHER_GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
OPENWEATHER_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"


@dataclass(frozen=True)
class CityProfile:
    """Static urban descriptors for a city used alongside weather data."""

    name: str
    lat: float
    lon: float
    population_density: int
    green_cover: float
    built_up_index: float
    country: str = "IN"


DEFAULT_CITY_PROFILES: List[CityProfile] = [
    CityProfile("Delhi", 28.6139, 77.2090, 12500, 0.14, 0.81),
    CityProfile("Ahmedabad", 23.0225, 72.5714, 10800, 0.11, 0.79),
    CityProfile("Jaipur", 26.9124, 75.7873, 7900, 0.16, 0.72),
    CityProfile("Nagpur", 21.1458, 79.0882, 7200, 0.18, 0.68),
    CityProfile("Hyderabad", 17.3850, 78.4867, 10200, 0.17, 0.75),
    CityProfile("Bengaluru", 12.9716, 77.5946, 12100, 0.23, 0.66),
    CityProfile("Kolkata", 22.5726, 88.3639, 24000, 0.09, 0.83),
    CityProfile("Mumbai", 19.0760, 72.8777, 20600, 0.10, 0.86),
]


def _to_nasa_date_string(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def fetch_nasa_power_city_data(
    city: CityProfile,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch daily public weather data for one city from NASA POWER."""

    end_date = end_date or date.today().isoformat()
    client = session or requests.Session()
    params = {
        "parameters": ",".join(
            [
                "T2M_MAX",
                "T2M_MIN",
                "T2M",
                "RH2M",
                "WS2M",
                "PS",
                "PRECTOTCORR",
                "ALLSKY_SFC_SW_DWN",
            ]
        ),
        "community": "RE",
        "longitude": city.lon,
        "latitude": city.lat,
        "start": _to_nasa_date_string(start_date),
        "end": _to_nasa_date_string(end_date),
        "format": "JSON",
    }
    response = client.get(NASA_POWER_DAILY_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    parameter_block = payload["properties"]["parameter"]
    raw_frame = pd.DataFrame(parameter_block)
    raw_frame.index = pd.to_datetime(raw_frame.index, format="%Y%m%d")
    raw_frame = raw_frame.rename(
        columns={
            "T2M_MAX": "temperature_max",
            "T2M_MIN": "temperature_min",
            "T2M": "temperature_avg",
            "RH2M": "humidity",
            "WS2M": "wind_speed",
            "PS": "air_pressure",
            "PRECTOTCORR": "precipitation",
            "ALLSKY_SFC_SW_DWN": "solar_radiation",
        }
    )
    raw_frame = raw_frame.replace(-999, np.nan)
    raw_frame = raw_frame.reset_index().rename(columns={"index": "date"})
    raw_frame["zone"] = city.name
    raw_frame["population_density"] = city.population_density
    raw_frame["green_cover"] = city.green_cover
    raw_frame["built_up_index"] = city.built_up_index
    raw_frame["latitude"] = city.lat
    raw_frame["longitude"] = city.lon
    return raw_frame


def build_public_climate_dataset(
    output_path: str | Path = "data/urban_heatwave_real_daily.csv",
    cities: Iterable[CityProfile] | None = None,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch and save a real public multi-city daily climate dataset."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cities = list(cities or DEFAULT_CITY_PROFILES)

    with requests.Session() as session:
        frames = [
            fetch_nasa_power_city_data(city, start_date=start_date, end_date=end_date, session=session)
            for city in cities
        ]

    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def get_city_profile(city_name: str) -> CityProfile | None:
    """Return a built-in city profile when available."""

    for profile in DEFAULT_CITY_PROFILES:
        if profile.name.lower() == city_name.lower():
            return profile
    return None


def geocode_city(city_name: str, api_key: str, limit: int = 1) -> Dict:
    """Resolve city coordinates using OpenWeatherMap geocoding."""

    params = {"q": city_name, "limit": limit, "appid": api_key}
    response = requests.get(OPENWEATHER_GEOCODE_URL, params=params, timeout=30)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"No coordinates found for city '{city_name}'.")
    return results[0]


def fetch_openweather_forecast(lat: float, lon: float, api_key: str) -> Dict:
    """Fetch current conditions and daily forecasts from OpenWeatherMap One Call 3.0."""

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "exclude": "minutely,hourly,alerts",
    }
    response = requests.get(OPENWEATHER_ONECALL_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def build_openweather_feature_payload(
    city_name: str,
    api_key: str | None = None,
    horizon_days: int = 3,
    population_density: int | None = None,
    green_cover: float | None = None,
    built_up_index: float | None = None,
) -> Dict[str, float | str]:
    """Aggregate OpenWeatherMap forecast data into the model's feature schema."""

    resolved_api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenWeatherMap API key missing. Set OPENWEATHERMAP_API_KEY or pass api_key explicitly.")

    built_in_profile = get_city_profile(city_name)
    geo = geocode_city(city_name, resolved_api_key)
    forecast = fetch_openweather_forecast(geo["lat"], geo["lon"], resolved_api_key)

    daily = forecast.get("daily", [])
    if not daily:
        raise ValueError("OpenWeatherMap daily forecast is missing from the response.")

    selected_days = daily[: max(1, min(horizon_days, len(daily)))]

    avg_temp = sum(day["temp"]["day"] for day in selected_days) / len(selected_days)
    humidity = sum(day["humidity"] for day in selected_days) / len(selected_days)
    wind_speed = sum(day["wind_speed"] for day in selected_days) / len(selected_days)
    air_pressure = sum(day["pressure"] for day in selected_days) / len(selected_days)
    precipitation = sum(day.get("rain", 0.0) for day in selected_days)
    solar_radiation_proxy = sum(day.get("uvi", 0.0) * 25 for day in selected_days) / len(selected_days)

    return {
        "zone": geo["name"],
        "temperature_max": max(day["temp"]["max"] for day in selected_days),
        "temperature_min": min(day["temp"]["min"] for day in selected_days),
        "temperature_avg": avg_temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "air_pressure": air_pressure,
        "precipitation": precipitation,
        "solar_radiation": solar_radiation_proxy,
        "population_density": population_density
        if population_density is not None
        else (built_in_profile.population_density if built_in_profile else 10000),
        "green_cover": green_cover if green_cover is not None else (built_in_profile.green_cover if built_in_profile else 0.15),
        "built_up_index": built_up_index
        if built_up_index is not None
        else (built_in_profile.built_up_index if built_in_profile else 0.75),
        "latitude": geo["lat"],
        "longitude": geo["lon"],
    }
