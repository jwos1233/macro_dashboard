#!/usr/bin/env python3
"""
Axe List Generator Module for Macro Dashboard
Handles crypto token analysis and ranking based on performance metrics
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime, timedelta

from config import COINGECKO_URL, BINANCE_URL, DEFAULT_TOP_N_TOKENS

class AxeListGenerator:
    """Generates ranked lists of crypto tokens based on performance metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.coingecko_url = COINGECKO_URL
        self.binance_url = BINANCE_URL
        self.last_baseline = None
        
        # Initialize session for API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Configuration
        self.config = config or {}
        self.min_market_cap = self.config.get('min_market_cap', 100_000_000)  # $100M
        self.min_volume = self.config.get('min_volume', 10_000_000)  # $10M
    
    def fetch_top_tokens(self, limit: int = 100) -> Optional[List[Dict]]:
        """Fetch top tokens from CoinGecko"""
        try:
            url = f"{self.coingecko_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            tokens = response.json()
            return [token for token in tokens if token['market_cap'] >= self.min_market_cap]
            
        except Exception as e:
            st.error(f"Failed to fetch top tokens: {e}")
            return None
    
    def fetch_token_performance(self, token_id: str, days: List[int] = [1, 7, 30]) -> Optional[Dict]:
        """Fetch performance data for a specific token"""
        try:
            url = f"{self.coingecko_url}/coins/{token_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': max(days)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            prices = data['prices']
            
            # Calculate returns for different periods
            performance = {}
            current_price = prices[-1][1]
            
            for day in days:
                if len(prices) > day:
                    past_price = prices[-(day + 1)][1]
                    return_pct = ((current_price - past_price) / past_price) * 100
                    performance[f'{day}d_return'] = return_pct
                else:
                    performance[f'{day}d_return'] = 0
            
            return performance
            
        except Exception as e:
            st.warning(f"Failed to fetch performance for {token_id}: {e}")
            return None
    
    def calculate_metrics(self, tokens: List[Dict]) -> pd.DataFrame:
        """Calculate performance metrics for all tokens"""
        if not tokens:
            return pd.DataFrame()
        
        # Prepare data for analysis
        token_data = []
        
        for token in tokens:
            try:
                # Fetch performance data
                performance = self.fetch_token_performance(token['id'])
                if not performance:
                    continue
                
                # Calculate additional metrics
                market_cap = token['market_cap']
                volume = token['total_volume']
                price_change_24h = token['price_change_percentage_24h']
                
                # Volume to market cap ratio
                volume_mc_ratio = volume / market_cap if market_cap > 0 else 0
                
                # Momentum score (weighted combination of returns)
                momentum_score = (
                    performance.get('1d_return', 0) * 0.3 +
                    performance.get('7d_return', 0) * 0.4 +
                    performance.get('30d_return', 0) * 0.3
                )
                
                token_data.append({
                    'id': token['id'],
                    'symbol': token['symbol'].upper(),
                    'name': token['name'],
                    'price': token['current_price'],
                    'market_cap': market_cap,
                    'volume': volume,
                    'volume_mc_ratio': volume_mc_ratio,
                    'price_change_24h': price_change_24h,
                    'day_return': performance.get('1d_return', 0),
                    'week_return': performance.get('7d_return', 0),
                    'month_return': performance.get('30d_return', 0),
                    'momentum_score': momentum_score
                })
                
            except Exception as e:
                st.warning(f"Error processing {token.get('symbol', 'Unknown')}: {e}")
                continue
        
        if not token_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(token_data)
        
        # Sort by momentum score
        df = df.sort_values('momentum_score', ascending=False)
        
        return df
    
    def run_analysis(self, top_n: int = DEFAULT_TOP_N_TOKENS) -> Optional[pd.DataFrame]:
        """Run complete axe list analysis"""
        try:
            with st.spinner("Fetching top tokens..."):
                tokens = self.fetch_top_tokens(limit=200)  # Fetch more to filter
                
            if not tokens:
                st.error("Failed to fetch token data")
                return None
            
            with st.spinner("Calculating performance metrics..."):
                df = self.calculate_metrics(tokens)
                
            if df.empty:
                st.error("Failed to calculate metrics")
                return None
            
            # Filter and return top N tokens
            result = df.head(top_n).copy()
            
            # Store baseline asset for reference
            if not result.empty:
                self.last_baseline = result.iloc[0]['symbol']
            
            return result
            
        except Exception as e:
            st.error(f"Error in axe list analysis: {e}")
            return None
    
    def get_analysis_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of axe list analysis"""
        if df.empty:
            return {}
        
        summary = {
            'total_tokens': len(df),
            'avg_momentum_score': df['momentum_score'].mean(),
            'top_performer': df.iloc[0]['symbol'],
            'top_momentum_score': df.iloc[0]['momentum_score'],
            'avg_market_cap': df['market_cap'].mean(),
            'avg_volume': df['volume'].mean(),
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
