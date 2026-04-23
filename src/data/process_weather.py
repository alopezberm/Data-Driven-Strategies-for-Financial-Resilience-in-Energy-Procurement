"""
process_weather.py

Batch processing and aggregation engine for historical weather data.
Implements a dual-weighting strategy (Population vs. Surface Area) to 
generate a consolidated national dataset for energy market analysis.
"""

import pandas as pd
import holidays
from pathlib import Path

def aggregate_weather_batches(
    raw_weather_dir: Path,
    provinces_info_file: Path,
    output_file: Path
) -> None:
    print("Step 1: Loading raw meteorological batches...")
    try:
        batch_1 = list(raw_weather_dir.glob("*top0-10*.csv"))[0]
        batch_2 = list(raw_weather_dir.glob("*top11-51*.csv"))[0]
    except IndexError:
        print("❌ Error: Batch files not found.")
        return

    df_1 = pd.read_csv(batch_1)
    df_2 = pd.read_csv(batch_2)
    df_raw = pd.concat([df_1, df_2], ignore_index=True)
    if "city" in df_raw.columns: df_raw["prov"] = df_raw["city"]

    print(f"Step 2: Processing weights from {provinces_info_file.name}...")
    # CAMBIO 1: El nuevo CSV usa comas (sep=',') y codificación estándar
    df_weights = pd.read_csv(provinces_info_file, sep=',')
    
    # CAMBIO 2: Los datos ya están limpios y la columna se llama 'prov'
    provinces_map = df_weights[['prov', '%Poblacion', '%Superficie2']].copy()
    provinces_map = provinces_map.rename(
        columns={'%Poblacion': 'pop_weight', '%Superficie2': 'surf_weight'}
    )

    print("Step 3: Applying Dual-Weighting Strategy...")
    df_merged = pd.merge(df_raw, provinces_map, on="prov", how="left")
    df_merged['date'] = pd.to_datetime(df_merged['date'], utc=True).dt.tz_localize(None)

    pop_cols = ['weather_code', 'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_mean']
    surf_cols = ['wind_speed_10m_max', 'wind_gusts_10m_max', 'shortwave_radiation_sum', 'precipitation_sum', 
                 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'surface_pressure_mean', 'et0_fao_evapotranspiration_sum']

    for c in pop_cols: df_merged[f'{c}_w'] = df_merged[c] * df_merged['pop_weight']
    for c in surf_cols: df_merged[f'{c}_w'] = df_merged[c] * df_merged['surf_weight']

    agg_map = {f'{c}_w': 'sum' for c in pop_cols + surf_cols}
    agg_map.update({'pop_weight': 'sum', 'surf_weight': 'sum'})
    df_agg = df_merged.groupby('date').agg(agg_map)

    for c in pop_cols: df_agg[c] = df_agg[f'{c}_w'] / df_agg['pop_weight']
    for c in surf_cols: df_agg[c] = df_agg[f'{c}_w'] / df_agg['surf_weight']

    # Conservamos nuestra valiosa desviación estándar para el EDA
    df_agg['std_avg_temperature'] = df_merged.groupby('date')['temperature_2m_mean'].std()

    print("Step 4: Engineering advanced features...")
    mad_data = df_merged[df_merged['prov'].str.contains('Madrid', case=False)].set_index('date')
    df_agg['sunrise'], df_agg['sunset'] = mad_data['sunrise'], mad_data['sunset']

    islands = ['Canarias', 'Las Palmas', 'Santa Cruz de Tenerife', 'Baleares', 'Illes Balears', 'Ceuta', 'Melilla']
    pen_data = df_merged[~df_merged['prov'].isin(islands)]
    df_agg['peninsular_max_temp'] = pen_data.groupby('date')['temperature_2m_max'].apply(lambda x: x.nlargest(3).min())
    df_agg['peninsular_min_temp'] = pen_data.groupby('date')['temperature_2m_min'].apply(lambda x: x.nsmallest(3).max())
    
    df_agg = df_agg.reset_index()
    es_holidays = holidays.Spain(years=range(2020, 2027))
    df_agg['is_national_holiday'] = df_agg['date'].dt.date.apply(lambda x: 1 if x in es_holidays else 0)

    # Limpieza de columnas temporales
    df_agg = df_agg.drop(columns=[c for c in df_agg.columns if c.endswith('_w')] + ['pop_weight', 'surf_weight'])
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_file, index=False)
    print(f"✅ SUCCESS! File generated: {output_file}")