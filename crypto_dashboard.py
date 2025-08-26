#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Updated with WoW quadrant analysis
Live dashboard with quadrant analysis, strategy performance, and week-over-week momentum tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import time
import requests

# Handle imports with proper error checking
YFINANCE_AVAILABLE = True
PLOTLY_AVAILABLE = True

try:
    import yfinance as yf
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings('ignore')

# ================================================================================================
# QUADRANT ANALYSIS MODULE
# ================================================================================================

@dataclass
class AssetClassification:
    symbol: str
    name: str
    primary_quadrant: str
    secondary_quadrant: str = None
    asset_type: str = ""
    weight: float = 1.0

class CurrentQuadrantAnalysis:
    def __init__(self, lookback_days=21):
        self.lookback_days = lookback_days
        self.asset_classifications = self._initialize_asset_classifications()
        self.quadrant_descriptions = {
            'Q1': 'Growth UP, Inflation DOWN (Goldilocks)',
            'Q2': 'Growth UP, Inflation UP (Reflation)', 
            'Q3': 'Growth DOWN, Inflation UP (Stagflation)',
            'Q4': 'Growth DOWN, Inflation DOWN (Deflation)'
        }
        self.core_assets = {
            'QQQ': 'NASDAQ 100 (Growth)', 'VUG': 'Vanguard Growth ETF',
            'IWM': 'Russell 2000 (Small Caps)', 'BTC-USD': 'Bitcoin (BTC)',
            'ETH-USD': 'Ethereum (ETH)',
            'XLE': 'Energy Sector ETF', 'DBC': 'Broad Commodities ETF',
            'GLD': 'Gold ETF', 'LIT': 'Lithium & Battery Tech ETF',
            'TLT': '20+ Year Treasury Bonds', 'XLU': 'Utilities Sector ETF',
            'VIXY': 'Short-Term VIX Futures ETF',
        }
    
    def _initialize_asset_classifications(self) -> Dict[str, AssetClassification]:
        classifications = {}
        q1_assets = [('QQQ', 'NASDAQ 100 (Growth)', 'Q1'), ('VUG', 'Vanguard Growth ETF', 'Q1'),
                     ('IWM', 'Russell 2000 (Small Caps)', 'Q1'), ('BTC-USD', 'Bitcoin (BTC)', 'Q1')]
        q2_assets = [('XLE', 'Energy Sector ETF', 'Q2'), ('DBC', 'Broad Commodities ETF', 'Q2')]
        q3_assets = [('GLD', 'Gold ETF', 'Q3'), ('LIT', 'Lithium & Battery Tech ETF', 'Q3')]
        q4_assets = [('TLT', '20+ Year Treasury Bonds', 'Q4'), ('XLU', 'Utilities Sector ETF', 'Q4'),
                     ('UUP', 'US Dollar Index ETF', 'Q4'), ('VIXY', 'Short-Term VIX Futures ETF', 'Q4')]
        
        for symbol, name, quad in q1_assets + q2_assets + q3_assets + q4_assets:
            classifications[symbol] = AssetClassification(symbol, name, quad)
        return classifications
    
    def fetch_recent_data(self, days_back=200):
        if not YFINANCE_AVAILABLE:
            st.error("Cannot fetch data: yfinance not available")
            return pd.DataFrame()
            
        data = {}
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for symbol in self.core_assets.keys():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 0:
                    clean_close = hist['Close'].dropna()
                    clean_close = clean_close[clean_close > 0]
                    if len(clean_close) >= 10:
                        data[symbol] = clean_close
            except Exception:
                continue
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.groupby(df.index.date).last()
            df.index = pd.to_datetime(df.index)
            df = df.fillna(method='ffill').dropna(how='all')
        return df
    
    def calculate_daily_momentum(self, price_data: pd.DataFrame) -> pd.DataFrame:
        momentum_data = pd.DataFrame(index=price_data.index)
        for symbol in price_data.columns:
            momentum_data[symbol] = price_data[symbol].pct_change(self.lookback_days) * 100
        return momentum_data
    
    def calculate_daily_quadrant_scores(self, momentum_data: pd.DataFrame) -> pd.DataFrame:
        quadrant_scores = pd.DataFrame(index=momentum_data.index)
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            quadrant_scores[f'{quad}_Score'] = 0.0
            quadrant_scores[f'{quad}_Count'] = 0
        
        for date in momentum_data.index:
            daily_scores = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
            daily_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
            
            for symbol in momentum_data.columns:
                if symbol in self.asset_classifications:
                    momentum = momentum_data.loc[date, symbol]
                    if pd.notna(momentum):
                        classification = self.asset_classifications[symbol]
                        quad = classification.primary_quadrant
                        weight = classification.weight
                        weighted_score = momentum * weight
                        daily_scores[quad] += weighted_score
                        daily_counts[quad] += 1
            
            for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
                quadrant_scores.loc[date, f'{quad}_Score'] = daily_scores[quad]
                quadrant_scores.loc[date, f'{quad}_Count'] = daily_counts[quad]
        
        return quadrant_scores
    
    def determine_daily_quadrant(self, quadrant_scores: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=quadrant_scores.index)
        
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            score_col = f'{quad}_Score'
            count_col = f'{quad}_Count'
            results[f'{quad}_Normalized'] = np.where(
                quadrant_scores[count_col] > 0,
                quadrant_scores[score_col] / quadrant_scores[count_col], 0)
        
        quad_cols = ['Q1_Normalized', 'Q2_Normalized', 'Q3_Normalized', 'Q4_Normalized']
        results['Primary_Quadrant'] = results[quad_cols].idxmax(axis=1).str.replace('_Normalized', '')
        results['Primary_Score'] = results[quad_cols].max(axis=1)
        
        results['Secondary_Score'] = results[quad_cols].apply(
            lambda row: row.nlargest(2).iloc[1] if len(row.nlargest(2)) > 1 else 0, axis=1)
        
        results['Confidence'] = np.where(results['Secondary_Score'] > 0,
                                       results['Primary_Score'] / results['Secondary_Score'], float('inf'))
        
        results['Regime_Strength'] = pd.cut(results['Confidence'],
                                          bins=[0, 1.2, 1.8, float('inf')],
                                          labels=['Weak', 'Medium', 'Strong'])
        
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            results[f'{quad}_Score'] = results[f'{quad}_Normalized']
        
        return results
    
    def calculate_quadrant_wow_changes(self, quadrant_scores: pd.DataFrame) -> pd.DataFrame:
        """Calculate week-over-week changes in quadrant scores"""
        
        # Get the last 14 days to calculate WoW change (7 days current vs 7 days prior)
        recent_scores = quadrant_scores.tail(14)
        
        if len(recent_scores) < 14:
            # If we don't have enough data, use available data
            if len(recent_scores) < 7:
                return pd.DataFrame()  # Not enough data
            
            # Use what we have - split in half
            mid_point = len(recent_scores) // 2
            current_week = recent_scores.iloc[mid_point:]
            prior_week = recent_scores.iloc[:mid_point]
        else:
            # Standard case: last 7 days vs prior 7 days
            current_week = recent_scores.tail(7)
            prior_week = recent_scores.iloc[-14:-7]
        
        wow_changes = []
        
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            score_col = f'{quad}_Score'
            
            # Calculate average scores for current and prior weeks
            current_avg = current_week[score_col].mean()
            prior_avg = prior_week[score_col].mean()
            
            # Calculate WoW change
            if prior_avg != 0:
                wow_change = ((current_avg - prior_avg) / abs(prior_avg)) * 100
            else:
                wow_change = 0 if current_avg == 0 else 100
            
            # Calculate absolute change for additional context
            absolute_change = current_avg - prior_avg
            
            wow_changes.append({
                'Quadrant': quad,
                'Description': self.quadrant_descriptions[quad],
                'Current_Avg': current_avg,
                'Prior_Avg': prior_avg,
                'WoW_Change_Pct': wow_change,
                'Absolute_Change': absolute_change,
                'Trend': 'Gaining' if wow_change > 1 else 'Losing' if wow_change < -1 else 'Stable'
            })
        
        # Convert to DataFrame and sort by absolute WoW change (descending)
        wow_df = pd.DataFrame(wow_changes)
        wow_df = wow_df.sort_values('WoW_Change_Pct', ascending=False)
        
        return wow_df

# ================================================================================================
# AXE LIST GENERATOR MODULE
# ================================================================================================

class AxeListGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.binance_url = "https://fapi.binance.com/fapi/v1"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.config = {
            'api_delay': 0.2, 'max_retries': 3, 'retry_backoff_base': 2,
            'default_top_n': 100, 'progress_interval': 10
        }
        if config:
            self.config.update(config)
    
    def get_top_tokens_by_market_cap(self, limit: int = 100, max_retries: Optional[int] = None) -> pd.DataFrame:
        if max_retries is None:
            max_retries = self.config['max_retries']
            
        for attempt in range(max_retries):
            try:
                url = f"{self.coingecko_url}/coins/markets"
                params = {
                    'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': limit,
                    'page': 1, 'sparkline': False, 'locale': 'en'
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                time.sleep(self.config['api_delay'])
                
                data = response.json()
                if not data:
                    st.error("No data returned from CoinGecko")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df = df[['id', 'symbol', 'name', 'market_cap', 'market_cap_rank', 'current_price']].copy()
                df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
                df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')
                df['market_cap_rank'] = pd.to_numeric(df['market_cap_rank'], errors='coerce')
                df = df.dropna(subset=['market_cap', 'current_price', 'market_cap_rank'])
                df = df.sort_values('market_cap_rank').head(limit)
                df['binance_symbol'] = df['symbol'].str.upper() + 'USDT'
                
                st.success(f"Fetched top {len(df)} tokens by market cap from CoinGecko")
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (self.config['retry_backoff_base'] ** attempt) * 5
                        st.warning(f"CoinGecko rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.error(f"CoinGecko rate limited after {max_retries} attempts")
                        return pd.DataFrame()
                else:
                    st.error(f"HTTP error when fetching from CoinGecko: {e}")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Error fetching from CoinGecko: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def validate_binance_symbols(self, tokens_df: pd.DataFrame) -> pd.DataFrame:
        st.info(f"Validating Binance symbols for {len(tokens_df)} tokens...")
        
        try:
            url = f"{self.binance_url}/exchangeInfo"
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.config['api_delay'])
            
            data = response.json()
            available_symbols = set()
            
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['contractType'] == 'PERPETUAL' and
                    symbol_info['quoteAsset'] == 'USDT'):
                    available_symbols.add(symbol_info['symbol'])
            
            st.info(f"Found {len(available_symbols)} available Binance USDT perpetual symbols")
            
            valid_tokens = []
            for _, token in tokens_df.iterrows():
                binance_symbol = token['binance_symbol']
                if binance_symbol in available_symbols:
                    valid_tokens.append(token)
                else:
                    alt_symbols = [
                        f"{token['symbol'].upper()}USDT",
                        f"{token['name'].upper().replace(' ', '')}USDT",
                        f"{token['id'].upper()}USDT"
                    ]
                    
                    for alt_symbol in alt_symbols:
                        if alt_symbol in available_symbols:
                            token['binance_symbol'] = alt_symbol
                            valid_tokens.append(token)
                            break
            
            result_df = pd.DataFrame(valid_tokens)
            st.success(f"{len(result_df)} tokens have valid Binance symbols")
            return result_df
            
        except Exception as e:
            st.warning(f"Error validating Binance symbols: {e}, using all tokens")
            return tokens_df
    
    def get_coin_data(self, symbol: str, days: int = 100, max_retries: Optional[int] = None) -> Optional[pd.DataFrame]:
        if max_retries is None:
            max_retries = self.config['max_retries']
            
        for attempt in range(max_retries):
            try:
                end_time = int(time.time() * 1000)
                start_time = end_time - (days * 24 * 60 * 60 * 1000)
                
                url = f"{self.binance_url}/klines"
                params = {
                    'symbol': symbol, 'interval': '1d', 'startTime': start_time,
                    'endTime': end_time, 'limit': 1000
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                time.sleep(self.config['api_delay'])
                
                data = response.json()
                if not data:
                    return None
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                if len(df) < days:
                    return None
                
                df['returns'] = df['price'].pct_change()
                df['ma_50'] = df['price'].rolling(window=50).mean()
                df['ma_20'] = df['price'].rolling(window=20).mean()
                df['above_ma50'] = df['price'] > df['ma_50']
                df['above_ma20'] = df['price'] > df['ma_20']
                df['ma50_distance'] = (df['price'] - df['ma_50']) / df['ma_50'] * 100
                
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (self.config['retry_backoff_base'] ** attempt) * 2
                        time.sleep(wait_time)
                        continue
                    else:
                        return None
                else:
                    return None
            except Exception:
                return None
        return None
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        if df is None or len(df) < 50:
            return None
        
        try:
            latest = df.iloc[-1]
            week_ago = df.iloc[-8] if len(df) >= 8 else df.iloc[0]
            month_ago = df.iloc[-31] if len(df) >= 31 else df.iloc[0]
            
            if pd.isna(latest['price']) or pd.isna(latest['ma_50']) or pd.isna(latest['ma_20']):
                return None
            
            return {
                'current_price': latest['price'],
                'ma_50': latest['ma_50'],
                'ma_20': latest['ma_20'],
                'above_ma50': latest['above_ma50'],
                'above_ma20': latest['above_ma20'],
                'ma50_distance': latest['ma50_distance'],
                'week_return': (latest['price'] / week_ago['price'] - 1) * 100,
                'month_return': (latest['price'] / month_ago['price'] - 1) * 100,
                'volatility': df['returns'].std() * np.sqrt(252) * 100,
                'sharpe_ratio': (df['returns'].mean() * 252) / (df['returns'].std() * np.sqrt(252)) if df['returns'].std() > 0 else 0
            }
        except Exception:
            return None
    
    def determine_baseline_asset(self) -> str:
        try:
            st.info("Analyzing BTCETH pair performance...")
            
            btc_data = self.get_coin_data('BTCUSDT', days=100)
            time.sleep(0.2)
            eth_data = self.get_coin_data('ETHUSDT', days=100)
            
            if btc_data is None or eth_data is None:
                st.warning("Failed to fetch BTC or ETH data, defaulting to BTC baseline")
                return 'BTC'
            
            btceth_ratio = btc_data['price'] / eth_data['price']
            ratio_ma_50 = btceth_ratio.rolling(window=50).mean()
            current_ratio = btceth_ratio.iloc[-1]
            current_ma_50 = ratio_ma_50.iloc[-1]
            
            btc_outperforming = current_ratio > current_ma_50
            
            if btc_outperforming:
                st.success("Baseline Asset: BTC (outperforming ETH)")
                self.last_baseline = 'BTC'
                return 'BTC'
            else:
                st.success("Baseline Asset: ETH (outperforming BTC)")
                self.last_baseline = 'ETH'
                return 'ETH'
                
        except Exception as e:
            st.error(f"Error determining baseline asset: {e}")
            return 'BTC'
    
    def calculate_ratio_ma_ranking(self, token_symbol: str, baseline_symbol: str) -> Optional[Dict]:
        try:
            token_data = self.get_coin_data(token_symbol, days=100)
            if token_data is None:
                return None
            
            baseline_data = self.get_coin_data(baseline_symbol, days=100)
            if baseline_data is None:
                return None
            
            if len(token_data) < 100 or len(baseline_data) < 100:
                return None
            
            if (pd.isna(token_data['price'].iloc[-1]) or pd.isna(baseline_data['price'].iloc[-1]) or
                pd.isna(token_data['ma_50'].iloc[-1]) or pd.isna(baseline_data['ma_50'].iloc[-1])):
                return None
            
            token_baseline_ratio = token_data['price'] / baseline_data['price']
            ratio_ma_50 = token_baseline_ratio.rolling(window=50).mean()
            current_ratio = token_baseline_ratio.iloc[-1]
            current_ma_50 = ratio_ma_50.iloc[-1]
            
            if pd.isna(current_ma_50):
                return None
            
            ratio_vs_ma = ((current_ratio - current_ma_50) / current_ma_50 * 100) if current_ma_50 > 0 else 0
            ratio_returns = token_baseline_ratio.pct_change().dropna()
            ratio_volatility = ratio_returns.std() * np.sqrt(252) if len(ratio_returns) > 0 else 0
            token_outperforming = current_ratio > current_ma_50
            
            return {
                'current_ratio': current_ratio,
                'ratio_ma_50': current_ma_50,
                'ratio_vs_ma': ratio_vs_ma,
                'ratio_volatility': ratio_volatility,
                'token_outperforming': token_outperforming,
                'ratio_strength_score': ratio_vs_ma
            }
        except Exception:
            return None
    
    def analyze_token_performance(self, symbol: str, baseline_asset: str) -> Optional[Dict]:
        try:
            token_data = self.get_coin_data(symbol, days=100)
            if token_data is None:
                return None
            
            time.sleep(self.config['api_delay'])
            
            baseline_symbol = 'BTCUSDT' if baseline_asset == 'BTC' else 'ETHUSDT'
            baseline_data = self.get_coin_data(baseline_symbol, days=100)
            
            if baseline_data is None:
                return None
            
            token_metrics = self.calculate_performance_metrics(token_data)
            baseline_metrics = self.calculate_performance_metrics(baseline_data)
            
            if token_metrics is None or baseline_metrics is None:
                return None
            
            correlation = 0
            beta = 0
            if len(token_data) >= 50 and len(baseline_data) >= 50:
                merged = pd.merge(token_data[['timestamp', 'returns']], 
                                baseline_data[['timestamp', 'returns']], 
                                on='timestamp', suffixes=('_token', '_baseline'))
                
                if len(merged) >= 30:
                    try:
                        clean_data = merged.dropna(subset=['returns_token', 'returns_baseline'])
                        if len(clean_data) >= 30:
                            correlation = clean_data['returns_token'].corr(clean_data['returns_baseline'])
                            covariance = clean_data['returns_token'].cov(clean_data['returns_baseline'])
                            baseline_variance = clean_data['returns_baseline'].var()
                            beta = covariance / baseline_variance if baseline_variance > 0 else 0
                    except Exception:
                        pass
            
            return {
                'symbol': symbol,
                'current_price': token_metrics['current_price'],
                'above_ma50': token_metrics['above_ma50'],
                'above_ma20': token_metrics['above_ma20'],
                'ma50_distance': token_metrics['ma50_distance'],
                'week_return': token_metrics['week_return'],
                'month_return': token_metrics['month_return'],
                'volatility': token_metrics['volatility'],
                'sharpe_ratio': token_metrics['sharpe_ratio'],
                'correlation_with_baseline': correlation,
                'beta_vs_baseline': beta,
                'relative_strength': token_metrics['month_return'] - baseline_metrics['month_return']
            }
        except Exception:
            return None
    
    def generate_axe_list(self, top_tokens: pd.DataFrame, baseline_asset: str) -> pd.DataFrame:
        st.info(f"Generating axe list based on {baseline_asset} baseline...")
        
        baseline_symbol = 'BTCUSDT' if baseline_asset == 'BTC' else 'ETHUSDT'
        analysis_results = []
        successful_analyses = 0
        failed_analyses = 0
        skipped_analyses = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (idx, token) in enumerate(top_tokens.iterrows()):
            binance_symbol = token['binance_symbol']
            name = token['name']
            
            if binance_symbol in ['BTCUSDT', 'ETHUSDT']:
                skipped_analyses += 1
                continue
            
            progress = (i + 1) / len(top_tokens)
            progress_bar.progress(progress)
            status_text.text(f"[{i+1}/{len(top_tokens)}] Analyzing {binance_symbol} ({name})...")
            
            analysis = self.analyze_token_performance(binance_symbol, baseline_asset)
            if analysis:
                ratio_analysis = self.calculate_ratio_ma_ranking(binance_symbol, baseline_symbol)
                
                if ratio_analysis:
                    analysis.update(ratio_analysis)
                else:
                    analysis.update({
                        'current_ratio': 0, 'ratio_ma_50': 0, 'ratio_vs_ma': 0,
                        'ratio_volatility': 0, 'token_outperforming': False, 'ratio_strength_score': 0
                    })
                
                analysis['name'] = name
                analysis['symbol'] = token['symbol']
                analysis['market_cap'] = token['market_cap']
                analysis['market_cap_rank'] = token['market_cap_rank']
                analysis_results.append(analysis)
                successful_analyses += 1
            else:
                failed_analyses += 1
            
            time.sleep(self.config['api_delay'])
        
        progress_bar.empty()
        status_text.empty()
        
        st.info(f"Analysis Complete: {successful_analyses} successful, {failed_analyses} failed, {skipped_analyses} skipped")
        
        if analysis_results:
            df = pd.DataFrame(analysis_results)
            
            df['axe_score'] = (
                df['above_ma50'].astype(int) * 2 +
                df['above_ma20'].astype(int) * 1 +
                (df['ma50_distance'] > 0).astype(int) * 1 +
                (df['week_return'] > 0).astype(int) * 1 +
                (df['month_return'] > 0).astype(int) * 1 +
                (df['relative_strength'] > 0).astype(int) * 2 +
                (df['correlation_with_baseline'] > 0.5).astype(int) * 1 +
                (df['beta_vs_baseline'] > 0.8).astype(int) * 1 +
                (df['token_outperforming']).astype(int) * 3 +
                (df['ratio_vs_ma'] > 0).astype(int) * 2
            )
            
            df = df.sort_values(['ratio_strength_score', 'axe_score'], ascending=[False, False])
            
            st.success(f"Generated axe list with {len(df)} tokens")
            return df
        else:
            st.error("No successful analyses to generate axe list")
            return pd.DataFrame()
    
    def run_analysis(self, top_n=None):
        if top_n is None:
            top_n = self.config.get('default_top_n', 100)
        
        st.info(f"Starting Axe List Analysis for top {top_n} tokens...")
        
        baseline = self.determine_baseline_asset()
        if not baseline:
            st.error("Failed to determine baseline asset")
            return None
        
        st.info("Fetching top tokens by market cap from CoinGecko...")
        top_tokens = self.get_top_tokens_by_market_cap(top_n)
        if top_tokens.empty:
            st.error("Failed to fetch top tokens")
            return None
        
        st.success(f"Found {len(top_tokens)} tokens to analyze")
        
        validated_tokens = self.validate_binance_symbols(top_tokens)
        if validated_tokens.empty:
            st.error("No valid Binance symbols found")
            return None
        
        axe_list = self.generate_axe_list(validated_tokens, baseline)
        if axe_list.empty:
            st.error("Failed to generate axe list")
            return None
        
        st.success(f"Analysis Complete! Successfully analyzed: {len(axe_list)} tokens")
        return axe_list

# ================================================================================================
# STRATEGY PERFORMANCE MODULE
# ================================================================================================

class StrategyPerformanceAnalysis:
    def __init__(self):
        self.strategy_name = "Quadrant Strategy"
        self.benchmark_name = "Buy & Hold"
    
    def calculate_strategy_performance(self, price_data: pd.DataFrame, daily_results: pd.DataFrame, crypto_symbol: str = 'BTC-USD', is_portfolio: bool = False) -> Dict:
        """Calculate performance metrics for quadrant strategy vs buy & hold"""
        
        if is_portfolio:
            if 'BTC-USD' not in price_data.columns or 'ETH-USD' not in price_data.columns:
                return None
        else:
            if price_data is None or crypto_symbol not in price_data.columns or daily_results is None:
                return None
        
        try:
            # Handle portfolio vs single crypto
            if is_portfolio:
                # Create 50/50 portfolio
                btc_prices = price_data['BTC-USD'].copy()
                eth_prices = price_data['ETH-USD'].copy()
                
                # Normalize to starting value of 100 for both
                btc_normalized = (btc_prices / btc_prices.iloc[0]) * 100
                eth_normalized = (eth_prices / eth_prices.iloc[0]) * 100
                
                # 50/50 portfolio
                crypto_prices = (btc_normalized + eth_normalized) / 2
                crypto_name = "50/50 BTC+ETH Portfolio"
            else:
                # Single crypto
                crypto_prices = price_data[crypto_symbol].copy()
                crypto_name = "Bitcoin" if crypto_symbol == 'BTC-USD' else "Ethereum"
            
            # Calculate 50 EMA
            crypto_50ema = crypto_prices.rolling(window=50).mean()
            
            # Forward fill quadrant data to match price data length
            aligned_quadrants = pd.Series('Q2', index=crypto_prices.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Forward fill quadrant assignments
            aligned_quadrants = aligned_quadrants.fillna(method='ffill')
            
            # CRITICAL FIX: Apply 1-day lag to avoid look-ahead bias
            # Shift quadrant signals forward by 1 day (lag the signal)
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            lagged_50ema = crypto_50ema.shift(1).fillna(crypto_prices.iloc[0])  # Lag EMA as well
            
            # Calculate daily returns
            crypto_returns = crypto_prices.pct_change().fillna(0)
            
            # Strategy: Long in Q1+Q3 AND above 50 EMA, Flat otherwise (using lagged signals)
            favorable_quadrant = lagged_quadrants.isin(['Q1', 'Q3'])
            above_ema = crypto_prices > lagged_50ema  # Current price vs lagged EMA
            strategy_positions = (favorable_quadrant & above_ema).astype(int)
            strategy_returns = crypto_returns * strategy_positions
            
            # Buy & Hold returns
            buyhold_returns = crypto_returns
            
            # Calculate cumulative performance
            strategy_cumulative = (1 + strategy_returns).cumprod()
            buyhold_cumulative = (1 + buyhold_returns).cumprod()
            
            # Performance metrics calculation
            def calculate_metrics(returns_series, name):
                if len(returns_series) == 0 or returns_series.std() == 0:
                    return {}
                
                total_return = (1 + returns_series).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
                volatility = returns_series.std() * np.sqrt(252)
                sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
                
                # Calculate drawdowns
                cumulative = (1 + returns_series).cumprod()
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = drawdowns.min()
                
                # Win rate (exclude flat periods for strategy)
                if name == self.strategy_name:
                    # Only count days when strategy was actually long
                    active_returns = returns_series[strategy_positions == 1]
                    positive_days = (active_returns > 0).sum()
                    total_days = len(active_returns)
                else:
                    positive_days = (returns_series > 0).sum()
                    total_days = len(returns_series[returns_series != 0])
                
                win_rate = positive_days / total_days if total_days > 0 else 0
                
                return {
                    'total_return': total_return * 100,
                    'annualized_return': annualized_return * 100,
                    'volatility': volatility * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown * 100,
                    'win_rate': win_rate * 100,
                    'cumulative_series': cumulative
                }
            
            strategy_metrics = calculate_metrics(strategy_returns, self.strategy_name)
            buyhold_metrics = calculate_metrics(buyhold_returns, self.benchmark_name)
            
            # Calculate strategy-specific metrics
            total_days = len(strategy_returns)
            long_days = strategy_positions.sum()
            flat_days = total_days - long_days
            time_in_market = (long_days / total_days) * 100 if total_days > 0 else 0
            
            # Calculate additional EMA-related metrics
            quadrant_only_positions = lagged_quadrants.isin(['Q1', 'Q3']).astype(int)
            quadrant_only_days = quadrant_only_positions.sum()
            ema_filter_reduction = ((quadrant_only_days - long_days) / quadrant_only_days * 100) if quadrant_only_days > 0 else 0
            
            return {
                'strategy_metrics': strategy_metrics,
                'buyhold_metrics': buyhold_metrics,
                'strategy_returns': strategy_returns,
                'buyhold_returns': buyhold_returns,
                'strategy_positions': strategy_positions,
                'aligned_quadrants': lagged_quadrants,  # Use lagged quadrants for display
                'time_in_market': time_in_market,
                'long_days': long_days,
                'flat_days': flat_days,
                'crypto_prices': crypto_prices,
                'crypto_50ema': lagged_50ema,
                'crypto_symbol': crypto_symbol,
                'crypto_name': crypto_name,
                'is_portfolio': is_portfolio,
                'btc_prices': price_data['BTC-USD'] if is_portfolio else None,
                'eth_prices': price_data['ETH-USD'] if is_portfolio else None,
                'quadrant_only_days': quadrant_only_days,
                'ema_filter_reduction': ema_filter_reduction,
                'signal_lag_applied': True,
                'ema_filter_applied': True  # Flag to indicate EMA filter is used
            }
            
        except Exception as e:
            st.error(f"Error calculating strategy performance: {e}")
            return None
    
    def create_performance_charts(self, performance_data: Dict):
        """Create performance visualization charts"""
        
        if not performance_data:
            st.error("No performance data available")
            return
        
        if not PLOTLY_AVAILABLE:
            st.error("Plotly required for strategy performance charts")
            return
        
        strategy_metrics = performance_data['strategy_metrics']
        buyhold_metrics = performance_data['buyhold_metrics']
        
        # Chart 1: Cumulative Performance Comparison
        fig_performance = go.Figure()
        
        # Strategy performance
        fig_performance.add_trace(go.Scatter(
            x=strategy_metrics['cumulative_series'].index,
            y=strategy_metrics['cumulative_series'].values * 100,  # Convert to percentage
            mode='lines',
            name=f"{self.strategy_name} ({performance_data['crypto_name']})",
            line=dict(color='#00ff00', width=2)
        ))
        
        # Buy & Hold performance
        fig_performance.add_trace(go.Scatter(
            x=buyhold_metrics['cumulative_series'].index,
            y=buyhold_metrics['cumulative_series'].values * 100,
            mode='lines',
            name=f"{self.benchmark_name} ({performance_data['crypto_name']})",
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_performance.update_layout(
            title="Cumulative Performance: Quadrant Strategy vs Buy & Hold",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Chart 2: Strategy Positions & Crypto Price
        fig_positions = go.Figure()
        
        # Crypto Price (secondary y-axis)
        crypto_name = performance_data['crypto_name']
        fig_positions.add_trace(go.Scatter(
            x=performance_data['crypto_prices'].index,
            y=performance_data['crypto_prices'].values,
            mode='lines',
            name=f'{crypto_name} Price',
            line=dict(color='orange', width=1),
            yaxis='y2'
        ))
        
        # Strategy positions
        positions_for_plot = performance_data['strategy_positions'] * performance_data['crypto_prices'].max() * 0.1
        fig_positions.add_trace(go.Scatter(
            x=performance_data['strategy_positions'].index,
            y=positions_for_plot.values,
            mode='lines',
            name='Long Positions',
            fill='tonexty',
            line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
            yaxis='y'
        ))
        
        fig_positions.update_layout(
            title=f"Strategy Positions vs {crypto_name} Price",
            xaxis_title="Date",
            yaxis=dict(title="Position Signal", side="left"),
            yaxis2=dict(title=f"{crypto_name} Price (USD)", side="right", overlaying="y"),
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_positions, use_container_width=True)

# ================================================================================================
# WOW ANALYSIS WIDGET
# ================================================================================================

def display_quadrant_wow_widget(analyzer: CurrentQuadrantAnalysis, quadrant_scores: pd.DataFrame):
    """Display the Quadrant WoW Changes widget"""
    
    st.subheader("Quadrant Momentum - Week over Week")
    
    wow_data = analyzer.calculate_quadrant_wow_changes(quadrant_scores)
    
    if wow_data.empty:
        st.warning("Insufficient data for WoW analysis")
        return
    
    # Create columns for the widget
    cols = st.columns(4)
    
    for i, (_, row) in enumerate(wow_data.iterrows()):
        with cols[i]:
            quadrant = row['Quadrant']
            wow_change = row['WoW_Change_Pct']
            trend = row['Trend']
            
            # Color coding based on change direction
            if wow_change > 2:
                card_color = "#d4edda"  # Light green
                border_color = "#28a745"  # Green
                arrow = "↗"
            elif wow_change < -2:
                card_color = "#f8d7da"  # Light red
                border_color = "#dc3545"  # Red
                arrow = "↘"
            else:
                card_color = "#fff3cd"  # Light yellow
                border_color = "#ffc107"  # Yellow
                arrow = "→"
            
            st.markdown(f'''
            <div style="
                background-color: {card_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin: 0.25rem;
            ">
                <h4>{quadrant} {arrow}</h4>
                <h3>{wow_change:+.1f}%</h3>
                <p style="font-size: 0.85rem; margin: 0;">
                    {trend}<br>
                    Avg: {row['Current_Avg']:.2f}
                </p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Detailed table below the cards
    with st.expander("Detailed WoW Analysis"):
        display_df = wow_data[['Quadrant', 'Description', 'WoW_Change_Pct', 'Current_Avg', 'Prior_Avg', 'Trend']].copy()
        display_df.columns = ['Quadrant', 'Description', 'WoW Change (%)', 'Current Avg', 'Prior Avg', 'Trend']
        display_df['WoW Change (%)'] = display_df['WoW Change (%)'].round(1)
        display_df['Current Avg'] = display_df['Current Avg'].round(2)
        display_df['Prior Avg'] = display_df['Prior Avg'].round(2)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Interpretation guide
    st.info("""
    **Interpretation Guide:**
    • **Green (↗)**: Quadrant gaining momentum (>+2% WoW)
    • **Red (↘)**: Quadrant losing momentum (<-2% WoW)  
    • **Yellow (→)**: Stable momentum (-2% to +2% WoW)
    • **Ranking**: Sorted by WoW change magnitude (highest to lowest)
    """)

# ================================================================================================
# STREAMLIT DASHBOARD
# ================================================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Crypto Macro Flow Dashboard",
        page_icon="charts", layout="wide", initial_sidebar_state="expanded")

    # Show dependency status
    if not YFINANCE_AVAILABLE:
        st.error("yfinance not available - Quadrant analysis disabled")
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available - Using basic charts")

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem; font-weight: bold; color: #1E88E5;
            text-align: center; margin-bottom: 2rem;
        }
        .quadrant-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem; border-radius: 10px; color: white;
            text-align: center; margin: 0.5rem 0;
        }
        .metric-card {
            background: #f8f9fa; padding: 1rem; border-radius: 8px;
            border-left: 4px solid #1E88E5; margin: 0.5rem 0;
        }
        .sidebar-info {
            background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.selectbox("Select Analysis", 
        ["Current Quadrant Analysis", "Strategy Performance", "Combined Dashboard"])

    # Settings
    st.sidebar.markdown("### Settings")
    lookback_days = st.sidebar.slider("Momentum Lookback (days)", 14, 50, 21)
    top_n_tokens = st.sidebar.slider("Top N Tokens for Axe List", 20, 100, 50)

    # Data loading function (removed caching to fix serialization error)
    def load_quadrant_data(lookback_days):
        if not YFINANCE_AVAILABLE:
            return None, None, None, None
            
        analyzer = CurrentQuadrantAnalysis(lookback_days=lookback_days)
        price_data = analyzer.fetch_recent_data(days_back=1095)
        if price_data.empty:
            return None, None, None, None
        
        momentum_data = analyzer.calculate_daily_momentum(price_data)
        quadrant_scores = analyzer.calculate_daily_quadrant_scores(momentum_data)
        daily_results = analyzer.determine_daily_quadrant(quadrant_scores)
        return price_data, daily_results, analyzer, quadrant_scores

    def create_charts(price_data, daily_results, analyzer):
        if price_data is None or 'BTC-USD' not in price_data.columns:
            st.error("BTC data not available")
            return
        
        # Prepare data for different time periods
        last_90_days = daily_results.tail(90)
        
        # Show current quadrant
        if not last_90_days.empty:
            latest_quad = last_90_days['Primary_Quadrant'].iloc[-1]
            if pd.notna(latest_quad):
                st.info(f"**Current Quadrant**: {latest_quad} - {analyzer.quadrant_descriptions[latest_quad]}")
        
        # Chart 1: Color-coded BTC Price Chart with 50 EMA Filter (3 years)
        st.subheader("Bitcoin Price (Last 3 Years)")
        
        if PLOTLY_AVAILABLE:
            # Use all available data for BTC chart (3 years)
            btc_data = price_data['BTC-USD'].copy()
            
            # Calculate 50 EMA
            btc_50ema = btc_data.rolling(window=50).mean()
            
            # Align quadrant data with BTC data (pad with Q2 for missing earlier data)
            aligned_quadrants = pd.Series('Q2', index=btc_data.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Create combined condition for green: (Q1 or Q3) AND above 50 EMA
            above_50ema = btc_data > btc_50ema
            favorable_quad = aligned_quadrants.isin(['Q1', 'Q3'])
            show_green = above_50ema & favorable_quad
            
            # Create plotly chart with color coding
            fig = go.Figure()
            
            # Add 50 EMA line first (behind the price)
            fig.add_trace(go.Scatter(
                x=btc_data.index,
                y=btc_50ema.values,
                mode='lines',
                line=dict(color='orange', width=1, dash='dash'),
                name='50 EMA',
                showlegend=True,
                opacity=0.7
            ))
            
            # Create segments for price line with different colors
            current_condition = show_green.iloc[0] if len(show_green) > 0 else False
            segment_start = 0
            
            for i in range(1, len(btc_data)):
                if show_green.iloc[i] != current_condition or i == len(btc_data) - 1:
                    # End of current segment
                    end_idx = i if i == len(btc_data) - 1 else i - 1
                    
                    # Determine color: Green only if in Q1/Q3 AND above 50 EMA, otherwise blue
                    color = '#00ff00' if current_condition else '#1f77b4'
                    
                    # Add line segment
                    fig.add_trace(go.Scatter(
                        x=btc_data.index[segment_start:end_idx+2],  # +2 to include next point
                        y=btc_data.values[segment_start:end_idx+2],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False  # Remove legend for price segments
                    ))
                    
                    # Start new segment
                    segment_start = i
                    current_condition = show_green.iloc[i]
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.info("Green: Q1/Q3 quadrants AND above 50 EMA | Blue: All other conditions | Orange dashed: 50 EMA")
            
        else:
            # Fallback to basic streamlit chart (3 years)
            btc_with_ema = pd.DataFrame({
                'BTC Price': price_data['BTC-USD'],
                '50 EMA': price_data['BTC-USD'].rolling(window=50).mean()
            })
            st.line_chart(btc_with_ema)
            st.info("Install plotly for color-coded quadrant chart with EMA filter")
        
        # Chart 2: Quadrant Scores (90 days only)
        st.subheader("Quadrant Scores - Last 90 Days")
        chart_data = last_90_days[['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score']]
        st.line_chart(chart_data)

    # Main content based on page selection
    if page == "Current Quadrant Analysis":
        st.markdown('<h1 class="main-header">Current Quadrant Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("Current Quadrant Analysis requires yfinance.")
            return
        
        # Load data
        with st.spinner("Loading quadrant analysis data..."):
            price_data, daily_results, analyzer, quadrant_scores = load_quadrant_data(lookback_days)
        
        if daily_results is not None:
            # Current quadrant info - changed to last 90 days
            last_90_days = daily_results.tail(90)
            current_data = last_90_days.iloc[-1]
            current_quadrant = current_data['Primary_Quadrant']
            current_score = current_data['Primary_Score']
            
            # Display current quadrant
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="quadrant-card">
                    <h3>Current Quadrant</h3>
                    <h1>{current_quadrant}</h1>
                    <p>{analyzer.quadrant_descriptions[current_quadrant]}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                prev_score = last_90_days['Primary_Score'].iloc[-2] if len(last_90_days) > 1 else current_score
                st.metric("Primary Score", f"{current_score:.2f}", 
                         delta=f"{current_score - prev_score:.2f}")
            
            with col3:
                confidence_val = current_data['Confidence']
                confidence_text = "Very High" if np.isinf(confidence_val) else f"{confidence_val:.2f}"
                st.metric("Confidence", confidence_text)
            
            with col4:
                st.metric("Regime Strength", current_data['Regime_Strength'])
            
            # WoW Analysis Widget - NEW
            st.markdown("---")
            display_quadrant_wow_widget(analyzer, quadrant_scores)
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                create_charts(price_data, daily_results, analyzer)
            
            with col2:
                # Additional info - updated for 90 days
                st.subheader("Quadrant Distribution (90 days)")
                recent_quads = last_90_days['Primary_Quadrant'].value_counts()
                st.bar_chart(recent_quads)
                
                # Color legend
                st.markdown("""
                **Chart Color Legend:**
                - Green: Q1 (Goldilocks) & Q3 (Stagflation)
                - Blue: Q2 (Reflation) & Q4 (Deflation)
                """)
            
            # 90-day table
            st.subheader("Last 90 Days Detailed View")
            
            display_df = last_90_days[['Primary_Quadrant', 'Primary_Score', 'Q1_Score', 
                                      'Q2_Score', 'Q3_Score', 'Q4_Score', 'Regime_Strength']].copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d')
            display_df.columns = ['Quadrant', 'Score', 'Q1', 'Q2', 'Q3', 'Q4', 'Strength']
            
            st.dataframe(display_df.round(2), use_container_width=True, height=400)
            
        else:
            st.error("Failed to load quadrant analysis data. Please check your internet connection.")

    elif page == "Strategy Performance":
        st.markdown('<h1 class="main-header">Strategy Performance Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("Strategy Performance Analysis requires yfinance.")
            return
        
        # Add crypto selection toggle
        st.sidebar.markdown("### Crypto Selection")
        crypto_choice = st.sidebar.radio(
            "Select Cryptocurrency",
            ("Bitcoin (BTC)", "Ethereum (ETH)", "50/50 BTC+ETH Portfolio"),
            help="Choose which cryptocurrency or portfolio to analyze with the quadrant strategy"
        )
        
        # Map choice to symbol and name
        if crypto_choice == "Bitcoin (BTC)":
            crypto_symbol = 'BTC-USD'
            crypto_name = "Bitcoin"
            is_portfolio = False
        elif crypto_choice == "Ethereum (ETH)":
            crypto_symbol = 'ETH-USD'
            crypto_name = "Ethereum"
            is_portfolio = False
        else:  # 50/50 Portfolio
            crypto_symbol = 'PORTFOLIO'
            crypto_name = "50/50 BTC+ETH Portfolio"
            is_portfolio = True
        
        # Load data
        with st.spinner("Loading strategy performance data..."):
            price_data, daily_results, analyzer, quadrant_scores = load_quadrant_data(lookback_days)
        
        if daily_results is not None and price_data is not None:
            # Check if required crypto data is available
            if is_portfolio:
                if 'BTC-USD' not in price_data.columns or 'ETH-USD' not in price_data.columns:
                    st.error("Both BTC and ETH data required for portfolio analysis.")
                    return
            else:
                if crypto_symbol not in price_data.columns:
                    st.error(f"{crypto_name} data not available in the dataset.")
                    return
                
            # Calculate strategy performance
            strategy_analyzer = StrategyPerformanceAnalysis()
            performance_data = strategy_analyzer.calculate_strategy_performance(price_data, daily_results, crypto_symbol, is_portfolio)
            
            if performance_data:
                strategy_metrics = performance_data['strategy_metrics']
                buyhold_metrics = performance_data['buyhold_metrics']
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Strategy Total Return", 
                             f"{strategy_metrics['total_return']:.1f}%",
                             delta=f"{strategy_metrics['total_return'] - buyhold_metrics['total_return']:.1f}% vs B&H",
                             help=f"Total return of quadrant strategy on {crypto_name} vs buy & hold")
                
                with col2:
                    st.metric("Sharpe Ratio", 
                             f"{strategy_metrics['sharpe_ratio']:.2f}",
                             delta=f"{strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']:.2f}",
                             help="Risk-adjusted return (higher is better)")
                
                with col3:
                    st.metric("Max Drawdown", 
                             f"{strategy_metrics['max_drawdown']:.1f}%",
                             delta=f"{strategy_metrics['max_drawdown'] - buyhold_metrics['max_drawdown']:.1f}%",
                             help="Worst peak-to-trough decline (lower is better)")
                
                with col4:
                    st.metric("Time in Market", 
                             f"{performance_data['time_in_market']:.1f}%",
                             delta=f"EMA Filter: -{performance_data['ema_filter_reduction']:.1f}%",
                             help="Percentage of time the strategy was long vs flat")
                
                # Performance charts
                strategy_analyzer.create_performance_charts(performance_data)
                
                # Detailed metrics comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quadrant Strategy Metrics")
                    strategy_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                                  'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
