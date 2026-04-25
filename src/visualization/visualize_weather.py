import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

def _prep_weather_data(df_weather_raw):
    """Prepara, renombra y filtra los datos meteorológicos para el periodo 2020-2025."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)
    
    df_w = df_weather_raw.copy()
    
    # 1. Renombrar columnas clave al formato estándar del EDA
    rename_mapping = {
        'date': 'Date',
        'temperature_2m_mean': 'temp',
        'shortwave_radiation_sum': 'solar_radiation',
        'wind_speed_10m_max': 'wind_speed'
    }
    # Solo renombramos las que existan en el df para evitar errores
    df_w.rename(columns={k: v for k, v in rename_mapping.items() if k in df_w.columns}, inplace=True)
    
    # 2. Asegurar que la fecha es Datetime y ordenar
    df_w['Date'] = pd.to_datetime(df_w['Date'])
    df_w.sort_values('Date', inplace=True)
    
    # 3. Filtrar horizonte temporal
    return df_w[(df_w['Date'] >= '2020-01-01') & (df_w['Date'] <= '2025-12-31')].copy()

def plot_thermal_profile(df_weather_raw):
    """2.4.1: Panel combinado: Estacionalidad macro vs Riesgo estructural (Extremos/Dispersión)."""
    df_w = _prep_weather_data(df_weather_raw)
    
    # Crear figura con 2 paneles (arriba tendencia, abajo extremos)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})

    # --- PANEL 1: ESTACIONALIDAD Y TENDENCIA (Media Móvil) ---
    ax1.plot(df_w['Date'], df_w['temp'], color='coral', alpha=0.3, label='Daily National Temp')
    ax1.plot(df_w['Date'], df_w['temp'].rolling(window=30).mean(), color='red', lw=2.5, label='30-Day Macro Trend (Seasonality)')
    ax1.set_title('Part A: The Expected Baseline (Macro Seasonality)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # --- PANEL 2: RIESGO DE COLA Y DISPERSIÓN (El "Flaw of Averages") ---
    ax2.plot(df_w['Date'], df_w['temp'], color='#1f77b4', linewidth=1.5, label='National Weighted Temp')
    
    if 'std_avg_temperature' in df_w.columns:
        ax2.fill_between(df_w['Date'], 
                         df_w['temp'] - df_w['std_avg_temperature'], 
                         df_w['temp'] + df_w['std_avg_temperature'], 
                         color='#1f77b4', alpha=0.15, label='Provincial Dispersion (±1 STD)')

    if 'peninsular_max_temp' in df_w.columns and 'peninsular_min_temp' in df_w.columns:
        ax2.plot(df_w['Date'], df_w['peninsular_max_temp'], color='darkred', alpha=0.6, linewidth=1.2, label='Peninsular Max (Stress Indicator)')
        ax2.plot(df_w['Date'], df_w['peninsular_min_temp'], color='darkblue', alpha=0.6, linewidth=1.2, label='Peninsular Min (Stress Indicator)')

    ax2.set_title('Part B: The Structural Risk (Spatial Dispersion & Extremes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # --- FORMATO GENERAL ---
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.suptitle('2.4.1 Comprehensive Thermal Profile (2020-2025)', fontweight='bold', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.show()

def plot_renewable_intermittency(df_weather_raw):
    """2.4.2: Visualiza la producción potencial de viento y sol."""
    df_w = _prep_weather_data(df_weather_raw)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Solar: Área rellena (radiación de onda corta)
    ax1.fill_between(df_w['Date'], 0, df_w['solar_radiation'], color='gold', alpha=0.4, label='Solar Radiation')
    # Viento: Línea suave (ráfagas máximas 10m, media móvil 7 días)
    ax2.plot(df_w['Date'], df_w['wind_speed'].rolling(7).mean(), color='skyblue', lw=2, label='Wind Speed (7d Avg Max)')

    ax1.set_ylabel('Shortwave Radiation Sum', color='orange', fontweight='bold')
    ax2.set_ylabel('Wind Speed Max (10m)', color='blue', fontweight='bold')
    plt.title('2.4.2 Renewable Intermittency: Solar & Wind Resource Potential', fontweight='bold')
    
    # Combinar leyendas
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_physical_financial_correlation(df_weather_raw, df_omip_raw):
    """2.4.3: Heatmap de correlación entre variables físicas y el precio Spot."""
    df_w = _prep_weather_data(df_weather_raw)
    
    # Preparar datos financieros
    df_f = df_omip_raw.copy()
    if 'Date' not in df_f.columns:
        if 'date' in df_f.columns:
            df_f.rename(columns={'date': 'Date'}, inplace=True)
        else:
            df_f.reset_index(inplace=True)
            if 'index' in df_f.columns:
                df_f.rename(columns={'index': 'Date'}, inplace=True)
                
    df_f['Date'] = pd.to_datetime(df_f['Date'])
    
    # Unimos para ver la correlación
    df_corr = pd.merge(df_w, df_f[['Date', 'Spot_Price_SPEL']], on='Date')
    
    # Seleccionamos variables clave para la matriz
    cols = ['Spot_Price_SPEL', 'temp', 'wind_speed', 'solar_radiation']
    cols = [c for c in cols if c in df_corr.columns]
    
    # Renombramos para que el gráfico quede más limpio
    display_names = {
        'Spot_Price_SPEL': 'Spot Price',
        'temp': 'Temperature',
        'wind_speed': 'Wind Speed',
        'solar_radiation': 'Solar Radiation'
    }
    df_corr.rename(columns=display_names, inplace=True)
    plot_cols = [display_names[c] for c in cols]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr[plot_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f",
                annot_kws={"size": 12}, linewidths=.5)
    plt.title('2.4.3 Physical-Financial Coupling: Weather vs. Spot Price', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()