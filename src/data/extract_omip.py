import pandas as pd
import requests
import time
import random
import io
from datetime import datetime, timedelta
from pathlib import Path

class OMIPScraper:
    def __init__(self):
        self.base_url_spot = "https://www.omip.pt/es/market-data/spot?commodity=EL&zone=ES&date="
        self.base_url_futures = "https://www.omip.pt/es/dados-mercado?product=EL&zone=ES&instrument=FTB&date="
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }

    def clean_val(self, val):
        if pd.isna(val) or str(val).lower() == 'n.a.' or str(val).strip() == '':
            return None
        return float(str(val).replace(',', ''))

    def get_daily_data(self, date_obj):
        date_str = date_obj.strftime('%Y-%m-%d')
        is_weekend = date_obj.weekday() >= 5 
        
        results = {"Date": date_str, "Spot_Price_SPEL": None}
        for i in range(1, 7):
            results[f"Future_M{i}_Price"] = None
            results[f"Future_M{i}_OpenInterest"] = None
        
        # 1. SPOT SCRAPING
        try:
            response_spot = requests.get(f"{self.base_url_spot}{date_str}", headers=self.headers, timeout=10)
            spot_tables = pd.read_html(io.StringIO(response_spot.text))
            
            for df in spot_tables:
                df_str = df.astype(str)
                if df_str.iloc[:, 0].str.contains('SPEL BASE', na=False).any():
                    row_idx = df_str[df_str.iloc[:, 0].str.contains('SPEL BASE', na=False)].index[0]
                    results['Spot_Price_SPEL'] = self.clean_val(df.iloc[row_idx, 1])
                    break
        except Exception as e:
            print(f"  [x] Spot Error on {date_str}: {e}")

        # 2. FUTURES SCRAPING
        if not is_weekend:
            try:
                response_fut = requests.get(f"{self.base_url_futures}{date_str}", headers=self.headers, timeout=10)
                future_tables = pd.read_html(io.StringIO(response_fut.text))
                
                for df in future_tables:
                    if df.shape[1] >= 15: 
                        df_str = df.astype(str)
                        future_rows = df_str[df_str.iloc[:, 0].str.contains('FTB M', na=False)].index.tolist()
                        
                        for idx, row_idx in enumerate(future_rows[:6]):
                            month_offset = idx + 1
                            results[f'Future_M{month_offset}_OpenInterest'] = self.clean_val(df.iloc[row_idx, 10])
                            results[f'Future_M{month_offset}_Price'] = self.clean_val(df.iloc[row_idx, 14])
                        
                        if future_rows:
                            break
            except Exception as e:
                print(f"  [x] Futures Error on {date_str}: {e}")
                
        return results

if __name__ == "__main__":
    # Define project structure paths automatically
    # This assumes the script is inside src/data/
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw" / "omip"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "omip_prices_raw.csv"

    # Define temporal scope
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31) # Adjust if needed
    current_date = start_date

    scraper = OMIPScraper()
    all_data = []

    print(f"Initiating Web Data Extraction from {start_date.date()} to {end_date.date()}...")

    while current_date <= end_date:
        day_data = scraper.get_daily_data(current_date)
        all_data.append(day_data)
        
        print(f"Processed: {current_date.date()} | Spot: {day_data['Spot_Price_SPEL']} | M+1: {day_data.get('Future_M1_Price')}")
        
        time.sleep(random.uniform(1.0, 2.5))
        current_date += timedelta(days=1)

    # DataFrame creation and missing value imputation
    final_df = pd.DataFrame(all_data)
    cols_to_ffill = [col for col in final_df.columns if 'Future' in col]
    final_df[cols_to_ffill] = final_df[cols_to_ffill].ffill()

    # Save to data/raw/omip/
    final_df.to_csv(output_file, index=False)
    print(f"\nData extraction completed successfully. File saved at: {output_file}")