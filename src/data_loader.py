import yfinance as yf
import pandas as pd
import numpy as np
import os
from typing import cast

class DataLoader:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.cache_path = f"data/{ticker}_{start_date}_{end_date}.parquet"

    def get_data(self) -> pd.DataFrame:
        """Fetch data from yfinance or local cache."""
        if os.path.exists(self.cache_path):
            print(f"Loading {self.ticker} from cache...")
            return pd.read_parquet(self.cache_path)
        
        print(f"Downloading {self.ticker} from yfinance...")
        raw_df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df = cast(pd.DataFrame, raw_df)
        
        if df.empty:
            raise ValueError("No data found. Check ticker or date range.")
            
        # Standardise: Use Close price and calculate Log Returns
        df = df[['Close', 'Volume']].copy()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Save to cache
        os.makedirs('data', exist_ok=True)
        df.to_parquet(self.cache_path)
        return df

if __name__ == "__main__":
    # Quick Test
    loader = DataLoader(ticker="SPY", start_date="2018-01-01", end_date="2024-01-01")
    data = loader.get_data()
    print(data.head())