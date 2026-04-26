import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_cointegration_and_oscillator(df_master: pd.DataFrame, zoom_start='2021-11-01', zoom_end='2022-03-31'):
    """
    Generates a 3-panel dashboard visualizing the cointegration between Spot and M1 Future prices,
    the distribution of their spread, and a zoomed-in 'Trading Oscillator' view of the actionable spread.
    
    Parameters:
    -----------
    df_master : pd.DataFrame
        The consolidated master dataset containing 'date', 'Spot_Price_SPEL', and 'Future_M1_Price'.
    zoom_start : str
        Start date for the Trading Oscillator zoom period (YYYY-MM-DD).
    zoom_end : str
        End date for the Trading Oscillator zoom period (YYYY-MM-DD).
    """
    # Create a copy to avoid mutating the original dataframe
    df = df_master.copy()
    
    # Set professional style
    sns.set_style("whitegrid")
    
    # 1. Calculate the Target Variable: The Spread
    df['Spread_Delta'] = df['Spot_Price_SPEL'] - df['Future_M1_Price']
    corr = df['Spot_Price_SPEL'].corr(df['Future_M1_Price'])
    
    # 2. Create a Dashboard Layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # --- Top Left: Cointegration Scatter ---
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(data=df, x='Future_M1_Price', y='Spot_Price_SPEL',
                scatter_kws={'alpha':0.2, 'color':'#2c3e50'}, 
                line_kws={'color':'darkred', 'linewidth': 2}, ax=ax1)
    ax1.set_title(f'Market Cointegration (r={corr:.2f})', fontweight='bold')
    ax1.set_xlabel('M1 Future Price (€/MWh)')
    ax1.set_ylabel('Spot Price (€/MWh)')
    
    # --- Top Right: Spread Distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df['Spread_Delta'], bins=50, kde=True, color='teal', ax=ax2)
    ax2.axvline(0, color='black', linestyle='--')
    ax2.set_title('Spread Distribution (Spot vs. M1)', fontweight='bold')
    ax2.set_xlabel('Delta (€/MWh)')
    ax2.set_ylabel('Frequency (Days)')
    
    # --- Bottom: The Trading Oscillator (Zoomed in for clarity) ---
    ax3 = fig.add_subplot(gs[1, :])
    
    # Ensure date is datetime for accurate filtering
    df['date'] = pd.to_datetime(df['date'])
    
    # Select a volatile period to show the AI's playground
    mask = (df['date'] >= pd.to_datetime(zoom_start)) & (df['date'] <= pd.to_datetime(zoom_end))
    df_zoom = df[mask]
    
    # Plot positive spread as red (spot is more expensive), negative as green (spot is cheaper)
    colors = np.where(df_zoom['Spread_Delta'] > 0, 'crimson', 'forestgreen')
    ax3.bar(df_zoom['date'], df_zoom['Spread_Delta'], color=colors, alpha=0.8, width=1.0)
    ax3.axhline(0, color='black', linewidth=1.5)
    
    ax3.set_title('AI Strategy View: The Actionable Spread Oscillator', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Price Delta (€/MWh) \n <-- Cheaper Spot | Expensive Spot -->')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    correlation = df_master['Spot_Price_SPEL'].corr(df_master['Future_M1_Price'])
    print(f"📊 Validated Pearson Correlation (Spot vs. M1 Future): {correlation:.3f}")

def plot_wind_impact(df):
    """
    Visualiza el impacto de la velocidad del viento en el precio Spot.
    Calcula e imprime la correlación de Pearson.
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot con línea de regresión para mostrar la tendencia
    sns.regplot(
        data=df, 
        x='wind_speed_10m_max', 
        y='Spot_Price_SPEL', 
        scatter_kws={'alpha':0.3, 'color': 'teal'},
        line_kws={'color':'red', 'linewidth': 2}
    )

    plt.title('Impact of Wind Speed on Electricity Spot Price', pad=15)
    plt.xlabel('Surface-Weighted Max Wind Speed (km/h)')
    plt.ylabel('OMIE Spot Price (€/MWh)')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Cálculo de Correlación de Pearson
    corr_wind = df['wind_speed_10m_max'].corr(df['Spot_Price_SPEL'])
    print(f"Pearson Correlation (Wind vs Price): {corr_wind:.3f}")


def plot_temperature_ucurve(df):
    """
    Visualiza la relación no lineal (curva en U) entre la temperatura y el precio.
    Utiliza una regresión polinómica de segundo orden.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot con regresión polinómica de orden 2
    sns.regplot(
        data=df, 
        x='temperature_2m_mean', 
        y='Spot_Price_SPEL', 
        order=2, # Captura la forma de U
        scatter_kws={'alpha':0.3, 'color': '#1f77b4'},
        line_kws={'color':'darkred', 'linewidth': 2}
    )

    plt.title('Impact of Temperature on Electricity Spot Price (U-Curve)', pad=15)
    plt.xlabel('Population-Weighted Mean Temperature (°C)')
    plt.ylabel('OMIE Spot Price (€/MWh)')
    plt.axvline(x=20, color='grey', linestyle='--', alpha=0.5, label='Thermal Comfort Zone (~20°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    corr_temp = df['Spot_Price_SPEL'].corr(df['temperature_2m_mean'])
    print(f"📊 Validated Pearson Correlation (Spot vs. Temperature): {corr_temp:.3f}")


def plot_state_space_heatmap(df):
    """
    Genera una matriz de correlación para las variables clave del espacio de estados.
    """
    # Selección de variables core
    core_features = [
        'Spot_Price_SPEL', 'Future_M1_Price', 
        'temperature_2m_mean', 'wind_speed_10m_max', 
        'shortwave_radiation_sum', 'precipitation_sum'
    ]

    # Calcular correlación
    corr_matrix = df[core_features].corr()

    # Dibujar Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        fmt=".2f", 
        linewidths=0.5
    )
    plt.title('State-Space Correlation Matrix', pad=15)
    plt.show()

# --- Ejemplo de uso en el Notebook ---
# from src.visualization.visualize_master_dataset import plot_wind_impact, plot_temperature_ucurve, plot_state_space_heatmap
# plot_wind_impact(df_master)
# plot_temperature_ucurve(df_master)
# plot_state_space_heatmap(df_master)