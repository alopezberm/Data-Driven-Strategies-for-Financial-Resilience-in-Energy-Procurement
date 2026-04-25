def generate_financial_eda_plots(df_raw):
    """
    Generates the 4 comprehensive Financial EDA plots for the technical report.
    Expects the raw OMIP dataframe as input.
    """
    # Professional Aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)

    # 0. Data Preparation
    df_eda = df_raw.copy()
    if 'Date' not in df_eda.columns:
        df_eda.reset_index(inplace=True)
    df_eda['Date'] = pd.to_datetime(df_eda['Date'])
    df_eda.sort_values('Date', inplace=True)
    
    # Filter for the 2020-2025 period
    df_eda = df_eda[(df_eda['Date'] >= '2020-01-01') & (df_eda['Date'] <= '2025-12-31')]

    # ==========================================================================
    # 2.3.1: MARKET REGIMES (CONTANGO VS BACKWARDATION)
    # ==========================================================================
    df_plot_1 = df_eda[(df_eda['Date'] >= '2021-06-01') & (df_eda['Date'] <= '2022-12-31')].copy()
    df_plot_1['Future_M1_Price'] = df_plot_1['Future_M1_Price'].ffill()

    plt.figure(figsize=(14, 5))
    plt.plot(df_plot_1['Date'], df_plot_1['Spot_Price_SPEL'], label='Spot Price', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(df_plot_1['Date'], df_plot_1['Future_M1_Price'], label='Future M+1', color='blue', linestyle='--', linewidth=2)

    plt.fill_between(df_plot_1['Date'], df_plot_1['Future_M1_Price'], df_plot_1['Spot_Price_SPEL'], 
                     where=(df_plot_1['Future_M1_Price'] >= df_plot_1['Spot_Price_SPEL']), 
                     interpolate=True, color='green', alpha=0.2, label='Contango (Future > Spot)')

    plt.fill_between(df_plot_1['Date'], df_plot_1['Future_M1_Price'], df_plot_1['Spot_Price_SPEL'], 
                     where=(df_plot_1['Future_M1_Price'] < df_plot_1['Spot_Price_SPEL']), 
                     interpolate=True, color='red', alpha=0.2, label='Backwardation (Spot > Future)')

    plt.title('2.3.1 Market Regimes: Identifying Structural Crises vs. Stability', fontweight='bold')
    plt.ylabel('Price (€/MWh)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # 2.3.2: OPERATIONAL RISK (VOLATILITY SIGNALS)
    # ==========================================================================
    df_eda['Spot_Vol_7d'] = df_eda['Spot_Price_SPEL'].rolling(window=7, min_periods=1).std()
    df_eda['Spot_Vol_30d'] = df_eda['Spot_Price_SPEL'].rolling(window=30, min_periods=1).std()

    df_vol_plot = df_eda[(df_eda['Date'] >= '2021-06-01') & (df_eda['Date'] <= '2022-12-31')]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

    ax1.plot(df_vol_plot['Date'], df_vol_plot['Spot_Price_SPEL'], color='black', alpha=0.8, linewidth=1.5, label='Spot Price (SPEL)')
    ax1.set_title('2.3.2 Operational Risk Assessment: Spot Price Dynamics and Volatility Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Absolute Spot Price\n(€/MWh)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.fill_between(df_vol_plot['Date'], df_vol_plot['Spot_Price_SPEL'], alpha=0.1, color='gray')

    ax2.plot(df_vol_plot['Date'], df_vol_plot['Spot_Vol_7d'], color='darkorange', linewidth=2, label='7-Day Volatility (Fast Signal)')
    ax2.plot(df_vol_plot['Date'], df_vol_plot['Spot_Vol_30d'], color='purple', linewidth=2.5, linestyle=':', label='30-Day Volatility (Structural Signal)')

    ax2.fill_between(df_vol_plot['Date'], df_vol_plot['Spot_Vol_7d'], df_vol_plot['Spot_Vol_30d'], 
                     where=(df_vol_plot['Spot_Vol_7d'] > df_vol_plot['Spot_Vol_30d']), 
                     interpolate=True, color='orange', alpha=0.2, label='Volatility Shock Emerging')

    ax2.set_ylabel('Rolling Std Dev\n(€/MWh)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # 2.3.3: MARKET LIQUIDITY (SPARSITY AUDIT)
    # ==========================================================================
    oi_cols = [col for col in df_eda.columns if 'OpenInterest' in col]
    if oi_cols:
        audit_data = []
        for col in oi_cols:
            maturity = col.split('_')[1] if len(col.split('_')) > 1 else col
            sparsity = ((df_eda[col].isna()) | (df_eda[col] == 0)).mean() * 100
            audit_data.append({
                'Maturity': maturity,
                'Median_Contracts': df_eda[col].median(),
                'Max_Contracts': df_eda[col].max(),
                'Sparsity (%)': round(sparsity, 2)
            })

        audit_df = pd.DataFrame(audit_data)

        plt.figure(figsize=(10, 4))
        sns.barplot(data=audit_df, x='Maturity', y='Sparsity (%)', color='steelblue')
        plt.axhline(y=50, color='red', linestyle='--', label='50% Sparsity Threshold (Cutoff)')
        plt.title('2.3.3 Market Liquidity: Sparsity of Forward Contracts', fontweight='bold')
        plt.ylabel('% of Days with Zero Liquidity')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("\n📊 Liquidity Audit Table:")
        display(audit_df)

    # ==========================================================================
    # 2.3.4: THE HEDGING OPPORTUNITY (M3 VARIANCE)
    # ==========================================================================
    df_plot_4 = df_eda.copy()
    df_plot_4.set_index('Date', inplace=True)
    
    if 'Future_M3_Price' in df_plot_4.columns:
        df_plot_4['M3_Variance'] = df_plot_4['Future_M3_Price'] - df_plot_4['Spot_Price_SPEL']
        
        plt.figure(figsize=(14, 4))
        plt.fill_between(df_plot_4.index, 0, df_plot_4['M3_Variance'], 
                         where=(df_plot_4['M3_Variance'] >= 0), color='green', alpha=0.3, 
                         label='Cost of Hedging (Future Premium)')
        plt.fill_between(df_plot_4.index, 0, df_plot_4['M3_Variance'], 
                         where=(df_plot_4['M3_Variance'] < 0), color='red', alpha=0.4, 
                         label='Opportunity Saved (Spot Spike)')
        plt.plot(df_plot_4.index, df_plot_4['M3_Variance'], color='black', lw=0.8, alpha=0.7)
        
        plt.title('2.3.4 The Hedging Opportunity: Anticipation Variance (M3 Future vs. Realized Spot)', fontweight='bold', fontsize=14)
        plt.ylabel('Variance / Margin (€/MWh)')
        plt.axhline(0, color='black', lw=1.5, linestyle='--')
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.show()