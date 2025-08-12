#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Final Working Version
Complete dashboard with quadrant analysis and axe list generator
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
            'Q1': 'Growth ↑, Inflation ↓ (Goldilocks)',
            'Q2': 'Growth ↑, Inflation ↑ (Reflation)', 
            'Q3': 'Growth ↓, Inflation ↑ (Stagflation)',
            'Q4': 'Growth ↓, Inflation ↓ (Deflation)'
        }
        self.core_assets = {
            'QQQ': 'NASDAQ 100 (Growth)', 'VUG': 'Vanguard Growth ETF',
            'IWM': 'Russell 2000 (Small Caps)', 'BTC-USD': 'Bitcoin (BTC)',
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
            st.error("❌ Cannot fetch data: yfinance not available")
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
                    st.error("❌ No data returned from CoinGecko")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df = df[['id', 'symbol', 'name', 'market_cap', 'market_cap_rank', 'current_price']].copy()
                df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
                df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')
                df['market_cap_rank'] = pd.to_numeric(df['market_cap_rank'], errors='coerce')
                df = df.dropna(subset=['market_cap', 'current_price', 'market_cap_rank'])
                df = df.sort_values('market_cap_rank').head(limit)
                df['binance_symbol'] = df['symbol'].str.upper() + 'USDT'
                
                st.success(f"✅ Fetched top {len(df)} tokens by market cap from CoinGecko")
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (self.config['retry_backoff_base'] ** attempt) * 5
                        st.warning(f"⚠️ CoinGecko rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.error(f"❌ CoinGecko rate limited after {max_retries} attempts")
                        return pd.DataFrame()
                else:
                    st.error(f"❌ HTTP error when fetching from CoinGecko: {e}")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"❌ Error fetching from CoinGecko: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def validate_binance_symbols(self, tokens_df: pd.DataFrame) -> pd.DataFrame:
        st.info(f"🔍 Validating Binance symbols for {len(tokens_df)} tokens...")
        
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
            st.success(f"✅ {len(result_df)} tokens have valid Binance symbols")
            return result_df
            
        except Exception as e:
            st.warning(f"⚠️ Error validating Binance symbols: {e}, using all tokens")
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
            st.info("🔍 Analyzing BTCETH pair performance...")
            
            btc_data = self.get_coin_data('BTCUSDT', days=100)
            time.sleep(0.2)
            eth_data = self.get_coin_data('ETHUSDT', days=100)
            
            if btc_data is None or eth_data is None:
                st.warning("❌ Failed to fetch BTC or ETH data, defaulting to BTC baseline")
                return 'BTC'
            
            btceth_ratio = btc_data['price'] / eth_data['price']
            ratio_ma_50 = btceth_ratio.rolling(window=50).mean()
            current_ratio = btceth_ratio.iloc[-1]
            current_ma_50 = ratio_ma_50.iloc[-1]
            
            btc_outperforming = current_ratio > current_ma_50
            
            if btc_outperforming:
                st.success("🎯 Baseline Asset: BTC (outperforming ETH)")
                return 'BTC'
            else:
                st.success("🎯 Baseline Asset: ETH (outperforming BTC)")
                return 'ETH'
                
        except Exception as e:
            st.error(f"❌ Error determining baseline asset: {e}")
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
        st.info(f"🔍 Generating axe list based on {baseline_asset} baseline...")
        
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
        
        st.info(f"📊 Analysis Complete: ✅ {successful_analyses} successful, ❌ {failed_analyses} failed, ⏭️ {skipped_analyses} skipped")
        
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
            
            st.success(f"🎯 Generated axe list with {len(df)} tokens")
            return df
        else:
            st.error("❌ No successful analyses to generate axe list")
            return pd.DataFrame()
    
    def run_analysis(self, top_n=None):
        if top_n is None:
            top_n = self.config.get('default_top_n', 100)
        
        st.info(f"🚀 Starting Axe List Analysis for top {top_n} tokens...")
        
        baseline = self.determine_baseline_asset()
        if not baseline:
            st.error("❌ Failed to determine baseline asset")
            return None
        
        st.info("🔍 Fetching top tokens by market cap from CoinGecko...")
        top_tokens = self.get_top_tokens_by_market_cap(top_n)
        if top_tokens.empty:
            st.error("❌ Failed to fetch top tokens")
            return None
        
        st.success(f"✅ Found {len(top_tokens)} tokens to analyze")
        
        validated_tokens = self.validate_binance_symbols(top_tokens)
        if validated_tokens.empty:
            st.error("❌ No valid Binance symbols found")
            return None
        
        axe_list = self.generate_axe_list(validated_tokens, baseline)
        if axe_list.empty:
            st.error("❌ Failed to generate axe list")
            return None
        
        st.success(f"🎯 Analysis Complete! Successfully analyzed: {len(axe_list)} tokens")
        return axe_list

# ================================================================================================
# STREAMLIT DASHBOARD
# ================================================================================================

def main():
    st.set_page_config(
        page_title="Crypto Macro Flow Dashboard",
        page_icon="📊", layout="wide", initial_sidebar_state="expanded")

    if not YFINANCE_AVAILABLE:
        st.error("❌ **yfinance not available** - Quadrant analysis disabled")
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ **Plotly not available** - Using basic charts")

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
    # Custom CSS styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2rem; 
            font-weight: bold; 
            color: #1E88E5;
            text-align: center; 
            margin-bottom: 2rem;
        }
        .quadrant-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem; 
            border-radius: 10px; 
            color: white;
            text-align: center; 
            margin: 0.5rem 0;
        }
        .metric-card {
            background: #f8f9fa; 
            padding: 1rem; 
            border-radius: 8px;
            border-left: 4px solid #1E88E5; 
            margin: 0.5rem 0;
        }
        .sidebar-info {
            background: #e3f2fd; 
            padding: 1rem; 
            border-radius: 8px; 
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.sidebar.title("🎯 Dashboard Controls")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.selectbox("📊 Select Analysis", 
        ["Current Quadrant Analysis", "Axe List Generator", "Combined Dashboard"])

    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    lookback_days = st.sidebar.slider("Momentum Lookback (days)", 14, 50, 21)
    top_n_tokens = st.sidebar.slider("Top N Tokens for Axe List", 20, 100, 50)

    # Data loading function
    def load_quadrant_data(lookback_days):
        if not YFINANCE_AVAILABLE:
            return None, None, None
            
        analyzer = CurrentQuadrantAnalysis(lookback_days=lookback_days)
        price_data = analyzer.fetch_recent_data(days_back=1095)
        if price_data.empty:
            return None, None, None
        
        momentum_data = analyzer.calculate_daily_momentum(price_data)
        quadrant_scores = analyzer.calculate_daily_quadrant_scores(momentum_data)
        daily_results = analyzer.determine_daily_quadrant(quadrant_scores)
        return price_data, daily_results, analyzer

    def create_charts(price_data, daily_results, analyzer):
        if price_data is None or 'BTC-USD' not in price_data.columns:
            st.error("BTC data not available")
            return
        
        btc_data = price_data[['BTC-USD']].copy()
        btc_data = btc_data.join(daily_results[['Primary_Quadrant']], how='left')
        
        st.subheader("Bitcoin Price (Last 3 Years)")
        st.line_chart(btc_data['BTC-USD'])
        
        if not btc_data.empty:
            latest_quad = btc_data['Primary_Quadrant'].iloc[-1]
            if pd.notna(latest_quad):
                st.info(f"**Current Quadrant**: {latest_quad} - {analyzer.quadrant_descriptions[latest_quad]}")
        
        if daily_results is not None:
            last_30_days = daily_results.tail(30)
            st.subheader("Quadrant Scores - Last 30 Days")
            chart_data = last_30_days[['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score']]
            st.line_chart(chart_data)

    # Main content based on page selection
    if page == "Current Quadrant Analysis":
        st.markdown('<h1 class="main-header">📊 Current Quadrant Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("❌ Current Quadrant Analysis requires yfinance.")
            return
        
        with st.spinner("Loading quadrant analysis data..."):
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
        if daily_results is not None:
            last_30_days = daily_results.tail(30)
            current_data = last_30_days.iloc[-1]
            current_quadrant = current_data['Primary_Quadrant']
            current_score = current_data['Primary_Score']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="quadrant-card">
                    <h3>🎯 Current Quadrant</h3>
                    <h1>{current_quadrant}</h1>
                    <p>{analyzer.quadrant_descriptions[current_quadrant]}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                prev_score = last_30_days['Primary_Score'].iloc[-2]
                st.metric("Primary Score", f"{current_score:.2f}", 
                         delta=f"{current_score - prev_score:.2f}")
            
            with col3:
                confidence_val = current_data['Confidence']
                confidence_text = "Very High" if np.isinf(confidence_val) else f"{confidence_val:.2f}"
                st.metric("Confidence", confidence_text)
            
            with col4:
                st.metric("Regime Strength", current_data['Regime_Strength'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                create_charts(price_data, daily_results, analyzer)
            
            with col2:
                st.subheader("📈 Recent Quadrant Trend")
                recent_quads = last_30_days['Primary_Quadrant'].value_counts()
                st.bar_chart(recent_quads)
            
            st.subheader("📈 Last 30 Days Detailed View")
            
            display_df = last_30_days[['Primary_Quadrant', 'Primary_Score', 'Q1_Score', 
                                      'Q2_Score', 'Q3_Score', 'Q4_Score', 'Regime_Strength']].copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d')
            display_df.columns = ['Quadrant', 'Score', 'Q1', 'Q2', 'Q3', 'Q4', 'Strength']
            
            st.dataframe(display_df.round(2), use_container_width=True)
            
        else:
            st.error("❌ Failed to load quadrant analysis data. Please check your internet connection.")

    elif page == "Axe List Generator":
        st.markdown('<h1 class="main-header">🎯 Axe List Generator</h1>', unsafe_allow_html=True)
        
        if st.button("🚀 Generate Axe List", type="primary"):
            try:
                generator = AxeListGenerator()
                axe_data = generator.run_analysis(top_n_tokens)
                
                if axe_data is not None and not axe_data.empty:
                    st.success(f"✅ Analysis complete! Found {len(axe_data)} tokens")
                    st.session_state['axe_data'] = axe_data
                else:
                    st.error("❌ Failed to generate axe list. Please try again.")
            except Exception as e:
                st.error(f"❌ Error generating axe list: {str(e)}")
        
        if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
            axe_data = st.session_state['axe_data']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Top Performer", axe_data.iloc[0]['name'])
            
            with col2:
                st.metric("Best Ratio vs MA", f"{axe_data['ratio_vs_ma'].max():.1f}%")
            
            with col3:
                outperforming = axe_data['token_outperforming'].sum()
                st.metric("Tokens Outperforming", f"{outperforming}/{len(axe_data)}")
            
            with col4:
                avg_return = axe_data['month_return'].mean()
                st.metric("Avg Monthly Return", f"{avg_return:.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10: Ratio vs 50-day MA")
                chart_data = axe_data.head(10).set_index('symbol')['ratio_vs_ma']
                st.bar_chart(chart_data)
            
            with col2:
                st.subheader("Week vs Month Returns")
                chart_data = axe_data[['week_return', 'month_return']].head(10)
                st.scatter_chart(chart_data)
            
            st.subheader("🏆 Top Performers Detailed View")
            
            display_cols = ['name', 'symbol', 'ratio_vs_ma', 'market_cap_rank', 
                           'week_return', 'month_return', 'above_ma50', 'above_ma20']
            display_df = axe_data[display_cols].copy()
            display_df['market_cap_rank'] = display_df['market_cap_rank'].astype(int)
            display_df.columns = ['Name', 'Symbol', 'Ratio vs MA (%)', 'MCap Rank', 
                                 'Week Return (%)', 'Month Return (%)', 'Above 50MA', 'Above 20MA']
            
            st.dataframe(display_df.round(2), use_container_width=True)

    else:  # Combined Dashboard
        st.markdown('<h1 class="main-header">🚀 Combined Macro Flow Dashboard</h1>', unsafe_allow_html=True)
        
        if YFINANCE_AVAILABLE:
            with st.spinner("Loading quadrant analysis..."):
                price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        else:
            price_data, daily_results, analyzer = None, None, None
        
        if daily_results is not None:
            current_data = daily_results.tail(30).iloc[-1]
            current_quadrant = current_data['Primary_Quadrant']
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown(f'''
                <div class="quadrant-card">
                    <h4>Current Regime</h4>
                    <h2>{current_quadrant}</h2>
                    <p>{analyzer.quadrant_descriptions[current_quadrant]}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
                    top_token = st.session_state['axe_data'].iloc[0]
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>🏆 Top Axe</h4>
                        <h3>{top_token['name']}</h3>
                        <p>Ratio vs MA: {top_token['ratio_vs_ma']:+.1f}%</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>🏆 Top Axe</h4>
                        <h3>Run Analysis</h3>
                        <p>Generate axe list first</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col3:
                if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
                    axe_data = st.session_state['axe_data']
                    outperforming = axe_data['token_outperforming'].sum()
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📊 Market Strength</h4>
                        <h3>{outperforming}/{len(axe_data)}</h3>
                        <p>Tokens outperforming baseline</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📊 Market Strength</h4>
                        <h3>-/-</h3>
                        <p>Generate axe list first</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if price_data is not None:
                    st.subheader("Bitcoin Price Trend")
                    st.line_chart(price_data['BTC-USD'].tail(365))
            
            with col2:
                if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
                    axe_data = st.session_state['axe_data']
                    st.subheader("Top 8 Tokens: Ratio vs MA")
                    chart_data = axe_data.head(8).set_index('symbol')['ratio_vs_ma']
                    st.bar_chart(chart_data)
                else:
                    st.info("Generate axe list to see top performers")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if daily_results is not None:
                    st.subheader("📈 Recent Quadrant Trend")
                    recent_quads = daily_results.tail(7)['Primary_Quadrant'].value_counts()
                    st.bar_chart(recent_quads)
            
            with col2:
                st.subheader("🎯 Top 5 Axe List")
                if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
                    axe_data = st.session_state['axe_data']
                    top_5 = axe_data.head(5)[['name', 'ratio_vs_ma', 'month_return']]
                    top_5.columns = ['Token', 'Ratio vs MA (%)', 'Month Return (%)']
                    st.dataframe(top_5.round(1), use_container_width=True)
                else:
                    st.info("Generate axe list to see top performers")
            
            if st.button("🚀 Generate/Refresh Axe List", type="primary"):
                with st.spinner("Generating axe list..."):
                    try:
                        generator = AxeListGenerator()
                        axe_data = generator.run_analysis(top_n_tokens)
                        
                        if axe_data is not None and not axe_data.empty:
                            st.session_state['axe_data'] = axe_data
                            st.success(f"✅ Axe list updated! Found {len(axe_data)} tokens")
                            st.experimental_rerun()
                        else:
                            st.error("❌ Failed to generate axe list")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
        else:
            st.error("❌ Failed to load dashboard data. Please refresh the page.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Sources")
    st.sidebar.markdown("• **Quadrant Analysis**: Yahoo Finance")
    st.sidebar.markdown("• **Axe List**: CoinGecko + Binance")
    st.sidebar.markdown("• **Refresh Rate**: 5 minutes")
    
    st.sidebar.markdown("### 🔧 System Status")
    if YFINANCE_AVAILABLE:
        st.sidebar.markdown("• ✅ Yahoo Finance: Ready")
    else:
        st.sidebar.markdown("• ❌ Yahoo Finance: Missing")
    
    if PLOTLY_AVAILABLE:
        st.sidebar.markdown("• ✅ Plotly Charts: Ready")
    else:
        st.sidebar.markdown("• ⚠️ Plotly Charts: Basic mode")
    
    with st.sidebar.expander("📖 How to Use"):
        st.markdown("""
        **Quadrant Analysis:**
        - Shows current market regime (Growth/Inflation)
        - BTC chart colored by quadrant periods
        - Last 30 days detailed scores
        
        **Axe List Generator:**
        - Finds tokens outperforming BTC/ETH baseline
        - Uses 50-day MA ratio analysis
        - Click 'Generate' to run analysis
        
        **Combined Dashboard:**
        - Overview of both analyses
        - Generate axe list for complete view
        """)

if __name__ == "__main__":
    main()
