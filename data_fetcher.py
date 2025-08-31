#!/usr/bin/env python3
"""
Data Fetcher Module for Macro Dashboard
Handles all data retrieval operations from various sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
import streamlit as st

# Handle imports with proper error checking
YFINANCE_AVAILABLE = True
try:
    import yfinance as yf
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

class DataFetcher:
    """Handles data fetching from various sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
    
    def fetch_recent_data(self, symbols: list, days_back: int = 200) -> pd.DataFrame:
        """
        Fetch recent price data for given symbols
        
        Args:
            symbols: List of symbol strings
            days_back: Number of days to look back
            
        Returns:
            DataFrame with price data indexed by date
        """
        if not YFINANCE_AVAILABLE:
            st.error("Cannot fetch data: yfinance not available")
            return pd.DataFrame()
            
        data = {}
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 0:
                    clean_close = hist['Close'].dropna()
                    clean_close = clean_close[clean_close > 0]
                    if len(clean_close) >= 10:
                        data[symbol] = clean_close
            except Exception as e:
                st.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.groupby(df.index.date).last()
            df.index = pd.to_datetime(df.index)
            df = df.fillna(method='ffill').dropna(how='all')
        return df
    
    def fetch_crypto_data(self, symbol: str, days_back: int = 200) -> Optional[pd.Series]:
        """
        Fetch data for a specific crypto symbol
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD')
            days_back: Number of days to look back
            
        Returns:
            Series with price data or None if failed
        """
        if not YFINANCE_AVAILABLE:
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days_back}d")
            if len(hist) > 0:
                return hist['Close'].dropna()
        except Exception as e:
            st.warning(f"Failed to fetch {symbol}: {e}")
            return None
        
        return None
    
    def calculate_momentum(self, price_data: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
        """
        Calculate momentum (percentage change) for price data
        
        Args:
            price_data: DataFrame with price data
            lookback_days: Number of days for momentum calculation
            
        Returns:
            DataFrame with momentum data
        """
        momentum_data = pd.DataFrame(index=price_data.index)
        for symbol in price_data.columns:
            momentum_data[symbol] = price_data[symbol].pct_change(lookback_days) * 100
        return momentum_data
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Drop any remaining NaN rows
        df = df.dropna(how='all')
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_symbols': len(df.columns),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d'),
                'days': len(df)
            },
            'missing_data': df.isnull().sum().to_dict(),
            'symbols': list(df.columns)
        }
        
        return summary
