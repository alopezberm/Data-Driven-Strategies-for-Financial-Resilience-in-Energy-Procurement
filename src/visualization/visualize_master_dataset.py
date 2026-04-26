import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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