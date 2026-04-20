"""
process_weather.py

Batch processing and aggregation engine for historical weather data.
Reads locally extracted Open-Meteo CSV batches and implements a dual-weighting 
strategy using the master Province Information CSV file.
"""

import pandas as pd
import holidays
from pathlib import Path

def aggregate_weather_batches(
    raw_weather_dir: Path,
    provinces_info_file: Path,
    output_file: Path
) -> None:
    print("Loading local raw CSV batches...")
    
    # 1. Locate the batch files
    try:
        file_batch_1 = list(raw_weather_dir.glob("*top0-10*.csv"))[0]
        file_batch_2 = list(raw_weather_dir.glob("*top11-51*.csv"))[0]
    except IndexError:
        print("❌ Error: Batch files not found in the raw data directory.")
        return

    df_1 = pd.read_csv(file_batch_1)
    df_2 = pd.read_csv(file_batch_2)
    df_all_weather = pd.concat([df_1, df_2], ignore_index=True)
    
    if "city" in df_all_weather.columns and "prov" not in df_all_weather.columns:
        df_all_weather["prov"] = df_all_weather["city"]

    # 2. Extract weights from the CSV
    print(f"Extracting weights from: {provinces_info_file.name}")
    df_weights = pd.read_csv(provinces_info_file, sep=';', encoding='latin-1')
    
    # Standardize numerical formats (handle '%' and European decimal commas)
    for col in ['%Poblacion', '%Superficie2']:
        df_weights[col] = (
            df_weights[col]
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
    
    # Rename columns for formal English internal logic
    provinces_data = df_weights[['Nombre', '%Poblacion', '%Superficie2']].copy()
    provinces_data = provinces_data.rename(columns={
        'Nombre': 'prov',
        '%Poblacion': 'population_weight',
        '%Superficie2': 'surface_weight'
    })

    # 3. Mathematical Aggregation
    print("Applying the Dual-Weighting Strategy...")
    df_merged = pd.merge(df_all_weather, provinces_data, on="prov", how="left")
    df_merged['date'] = pd.to_datetime(df_merged['date'], utc=True).dt.tz_localize(None)

    cols_pop = ['weather_code', 'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_mean']
    cols_surf = ['wind_speed_10m_max', 'wind_gusts_10m_max', 'shortwave_radiation_sum', 'precipitation_sum', 
                 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'surface_pressure_mean', 'et0_fao_evapotranspiration_sum']

    for col in cols_pop: 
        df_merged[f'{col}_w'] = df_merged[col] * df_merged['population_weight']
    for col in cols_surf: 
        df_merged[f'{col}_w'] = df_merged[col] * df_merged['surface_weight']

    agg_funcs = {f'{col}_w': 'sum' for col in cols_pop + cols_surf}
    agg_funcs.update({'population_weight': 'sum', 'surface_weight': 'sum'})

    df_final = df_merged.groupby('date').agg(agg_funcs)

    for col in cols_pop: 
        df_final[col] = df_final[f'{col}_w'] / df_final['population_weight']
    for col in cols_surf: 
        df_final[col] = df_final[f'{col}_w'] / df_final['surface_weight']

    df_final = df_final.drop(columns=[f'{col}_w' for col in cols_pop + cols_surf] + ['population_weight', 'surface_weight'])

    # 4. Feature Engineering
    print("Engineering advanced features...")
    df_madrid = df_merged[df_merged['prov'].str.contains('Madrid', case=False, na=False)].set_index('date')
    df_final['sunrise'] = df_madrid['sunrise']
    df_final['sunset'] = df_madrid['sunset']

    # Peninsular data to exclude island outliers
    non_peninsular = ['Canarias', 'Las Palmas', 'Santa Cruz de Tenerife', 'Baleares', 'Illes Balears', 'Ceuta', 'Melilla']
    df_peninsular = df_merged[~df_merged['prov'].isin(non_peninsular)]

    df_final['peninsular_max_temperature'] = df_peninsular.groupby('date')['temperature_2m_max'].apply(lambda x: x.nlargest(3).min())
    df_final['peninsular_min_temperature'] = df_peninsular.groupby('date')['temperature_2m_min'].apply(lambda x: x.nsmallest(3).max())
    df_final['max_windspeed'] = df_merged.groupby('date')['wind_speed_10m_max'].apply(lambda x: x.nlargest(3).min())
    df_final = df_final.reset_index()

    # Holidays
    es_holidays = holidays.Spain(years=range(2020, 2027))
    df_final['Is_national_holiday'] = df_final['date'].dt.date.apply(lambda x: 1 if x in es_holidays else 0)

    # 5. Export
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(f"✅ SUCCESS! National aggregated dataset exported to:\n{output_file}")