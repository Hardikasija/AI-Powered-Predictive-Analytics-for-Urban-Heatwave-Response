import os
import folium
import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None


def create_heat_risk_map(grid_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    center = [grid_df["lat"].mean(), grid_df["lon"].mean()]
    heat_map = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    for _, row in grid_df.iterrows():
        risk = row.get("heatwave_risk", 0)
        color = "green"
        if risk > 0.7:
            color = "red"
        elif risk > 0.4:
            color = "orange"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['grid_id']} risk={risk:.2f}",
        ).add_to(heat_map)

    path = os.path.join(output_dir, "heat_risk_map.html")
    heat_map.save(path)
    return path


def create_geojson_map(geojson_path: str, output_dir: str, risk_col: str = "heatwave_risk") -> str:
    if gpd is None:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    gdf = gpd.read_file(geojson_path)
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]

    heat_map = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    def style_function(feature):
        risk = feature["properties"].get(risk_col, 0)
        if risk > 0.7:
            color = "#d73027"
        elif risk > 0.4:
            color = "#fc8d59"
        else:
            color = "#91cf60"
        return {
            "fillColor": color,
            "color": "#444",
            "weight": 1,
            "fillOpacity": 0.6,
        }

    folium.GeoJson(gdf, style_function=style_function).add_to(heat_map)
    path = os.path.join(output_dir, "heat_risk_map_geojson.html")
    heat_map.save(path)
    return path
