import pandas as pd
import numpy as np

def build_domain_features(df_input: pd.DataFrame, base_temp: float = 20.0, seed: int = 42) -> pd.DataFrame:
    """
    Applies Domain-Driven Feature Engineering for the Iberian Energy Market.
    Includes deterministic foresight (D+1), stochastic weather forecasting,
    thermal distillation (HDD/CDD), and the financial spread.
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        The merged master dataset.
    base_temp : float
        The thermal comfort zone baseline (default: 20.0°C).
    seed : int
        Random seed for reproducible stochastic noise generation.
        
    Returns:
    --------
    pd.DataFrame
        Dataset enriched with domain-specific AI features.
    """
    df = df_input.copy()
    
    # Ensure date is datetime for calendar operations
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # -------------------------------------------------------------------------
    # 1. FINANCIAL IMPUTATION & SPREAD
    # -------------------------------------------------------------------------
    future_cols = [col for col in df.columns if 'Future' in col]
    df[future_cols] = df[future_cols].bfill().ffill()
    
    if 'Spot_Price_SPEL' in df.columns and 'Future_M1_Price' in df.columns:
        df['Spot_M1_Spread'] = df['Spot_Price_SPEL'] - df['Future_M1_Price']

    # -------------------------------------------------------------------------
    # 2. DETERMINISTIC MARKET INFO (The "Scraping" Advantage)
    # The auction for D+1 is public by 13:00 CET today.
    # -------------------------------------------------------------------------
    df['Spot_Price_SPEL_t+1_known'] = df['Spot_Price_SPEL'].shift(-1)

    # -------------------------------------------------------------------------
    # 3. THERMAL & RENEWABLE DISTILLATION
    # -------------------------------------------------------------------------
    temp_col = 'apparent_temperature_mean' if 'apparent_temperature_mean' in df.columns else 'temperature_2m_mean'
    if temp_col in df.columns:
        df['HDD'] = np.maximum(0, base_temp - df[temp_col])
        df['CDD'] = np.maximum(0, df[temp_col] - base_temp)

    if 'shortwave_radiation_sum' in df.columns:
        df['solar_intensity'] = df['shortwave_radiation_sum'] / (df['shortwave_radiation_sum'].max() + 1e-9)
        df['is_solar_peak'] = (df['shortwave_radiation_sum'] > 20000).astype(int)

    if 'wind_speed_10m_max' in df.columns:
        df['is_high_wind'] = (df['wind_speed_10m_max'] > 20).astype(int)

    # -------------------------------------------------------------------------
    # 4. STOCHASTIC WEATHER FORESIGHT (D+1 to D+3)
    # -------------------------------------------------------------------------
    np.random.seed(seed)
    forecast_targets = ['solar_intensity', 'HDD', 'CDD', 'is_high_wind']
    
    for col in forecast_targets:
        if col in df.columns:
            for day in [1, 2, 3]:
                future_real = df[col].shift(-day)
                # Noise level: 5% * day (D+1: 5%, D+2: 10%, D+3: 15%)
                noise = np.random.normal(0, future_real.std() * (0.05 * day), size=len(future_real))
                df[f'{col}_pred_t+{day}'] = future_real + noise

    # -------------------------------------------------------------------------
    # 5. CALENDAR & HOLIDAY STANDARDIZATION
    # -------------------------------------------------------------------------
    holiday_col = 'is_national_holiday' if 'is_national_holiday' in df.columns else 'is_holiday'
    if holiday_col in df.columns:
        df['is_holiday'] = df[holiday_col].fillna(0).astype(int)
        if holiday_col != 'is_holiday': 
            df = df.drop(columns=[holiday_col])
    else:
        df['is_holiday'] = 0

    if 'date' in df.columns:
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # -------------------------------------------------------------------------
    # 6. THE UNKNOWN HORIZON (TARGETS FOR THE AI)
    # -------------------------------------------------------------------------
    df['Spot_Price_target_t+2'] = df['Spot_Price_SPEL'].shift(-2)
    df['Spot_Price_target_t+3'] = df['Spot_Price_SPEL'].shift(-3)

    # -------------------------------------------------------------------------
    # 7. CLEANUP: Dropping raw features used for distillation
    # -------------------------------------------------------------------------
    raw_weather = [
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 
        'apparent_temperature_mean', 'wind_speed_10m_max', 'shortwave_radiation_sum', 
        'precipitation_sum', 'weather_code'
    ]
    df = df.drop(columns=[c for c in raw_weather if c in df.columns])

    return df