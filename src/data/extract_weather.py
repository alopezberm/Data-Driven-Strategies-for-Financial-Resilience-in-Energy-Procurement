"""
extract_weather.py

Pure extraction engine for historical weather data via Open-Meteo API.
This script fetches data in batches to respect API rate limits and exports 
the raw data to CSV files.
"""

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pathlib import Path

def extract_weather_batch(
    provinces_file: str | Path,
    output_dir: str | Path,
    num_init: int,
    num_end: int,
    start_date: str = "2020-01-01",
    end_date: str = "2026-03-24"
) -> None:
    """
    Extracts weather data for a specific batch of provinces.
    """
    print(f"Loading province metadata from: {provinces_file}")
    
    # 1. Load Province Coordinates
    df_provs = pd.read_excel(provinces_file, sheet_name="df_LatLon")
    df_provs["lat"] = df_provs["Lat"].astype(float)
    df_provs["lon"] = df_provs["Lon"].astype(float)
    df_provs["prov"] = df_provs["Nombre"].astype(str)
    
    # 2. Setup Open-Meteo API Client
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    city_names = df_provs.iloc[num_init:num_end+1]["prov"].to_list()
    
    params = {
        "latitude": df_provs.iloc[num_init:num_end+1]["lat"].to_list(),
        "longitude": df_provs.iloc[num_init:num_end+1]["lon"].to_list(),
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", 
                  "apparent_temperature_mean", "wind_speed_10m_max", "wind_gusts_10m_max", 
                  "shortwave_radiation_sum", "sunrise", "sunset", "daylight_duration", 
                  "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", 
                  "surface_pressure_mean", "et0_fao_evapotranspiration_sum"],
    }

    print(f"Fetching API data for cities index {num_init} to {num_end}...")
    responses = openmeteo.weather_api(url, params=params)
    
    city_frames = []
    
    # 3. Process the API responses
    for city_name, response in zip(city_names, responses):
        daily = response.Daily()
        
        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}
        
        for i, var_name in enumerate(params["daily"]):
            if var_name in ["sunrise", "sunset"]:
                daily_data[var_name] = daily.Variables(i).ValuesInt64AsNumpy()
            else:
                daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()
                
        daily_city_df = pd.DataFrame(data=daily_data)
        daily_city_df["city"] = city_name
        city_frames.append(daily_city_df)

    # 4. Concatenate and Export
    df_batch = pd.concat(city_frames, ignore_index=True)
    
    extract_date = pd.to_datetime("today").strftime('%Y-%m-%d')
    dates_from_to = f"{start_date[:4]}-{end_date[:4]}"
    csv_title = f"daily_top{num_init}-{num_end}_cities_{dates_from_to}_{extract_date}_extracted.csv"
    
    output_path = Path(output_dir) / csv_title
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_batch.to_csv(output_path, index=False)
    print(f"✅ Batch successfully saved to: {output_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROVINCES_FILE = PROJECT_ROOT / "data" / "external" / "_Provincias_Info.xlsx"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
    
    # extract_weather_batch(PROVINCES_FILE, OUTPUT_DIR, num_init=0, num_end=10)
    # extract_weather_batch(PROVINCES_FILE, OUTPUT_DIR, num_init=11, num_end=51)