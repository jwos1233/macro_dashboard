#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Complete Version with Current Quadrant Asset Analysis
Live dashboard with quadrant analysis, strategy performance, and current quadrant asset ranking
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
                self.last_baseline = 'BTC'
                return 'BTC'
            else:
                st.success("🎯 Baseline Asset: ETH (outperforming BTC)")
                self.last_baseline = 'ETH'
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
# STREAMLIT DASHBOARD
# ================================================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Crypto Macro Flow Dashboard",
        page_icon="📈", layout="wide", initial_sidebar_state="expanded")

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
        ["Current Quadrant Analysis", "Strategy Performance", "Current Quadrant Assets", "Combined Dashboard"])

    # Settings
    st.sidebar.markdown("### Settings")
    lookback_days = st.sidebar.slider("Momentum Lookback (days)", 14, 50, 21)
    top_n_tokens = st.sidebar.slider("Top N Tokens for Axe List", 20, 100, 50)

    # Data loading function (removed caching to fix serialization error)
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
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
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

    elif page == "Current Quadrant Assets":
        st.markdown('<h1 class="main-header">Current Quadrant Asset Performance</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("Current Quadrant Asset Analysis requires yfinance.")
            return
        
        # Load quadrant data
        with st.spinner("Loading quadrant and asset data..."):
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
        if daily_results is not None and price_data is not None:
            # Get current quadrant
            current_quadrant = daily_results.iloc[-1]['Primary_Quadrant']
            current_score = daily_results.iloc[-1]['Primary_Score']
            
            # Display current quadrant info
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f'''
                <div class="quadrant-card">
                    <h3>Current Market Regime</h3>
                    <h1>{current_quadrant}</h1>
                    <p>{analyzer.quadrant_descriptions[current_quadrant]}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.metric("Quadrant Score", f"{current_score:.2f}")
            
            with col3:
                # Count days in current quadrant in last 30 days
                last_30_days = daily_results.tail(30)
                current_quad_days = (last_30_days['Primary_Quadrant'] == current_quadrant).sum()
                st.metric("Days in Current Quad", f"{current_quad_days}/30")
            
            # APPLY SAME LAG LOGIC AS STRATEGY TO AVOID LOOK-AHEAD BIAS
            # Create lagged quadrant signals (shift by 1 day)
            aligned_quadrants = pd.Series('Q2', index=price_data.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Apply 1-day lag - we only know quadrant AFTER market close
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            
            # APPLY SAME LAG LOGIC AS STRATEGY TO AVOID LOOK-AHEAD BIAS  
            # Create lagged quadrant signals (shift by 1 day)
            aligned_quadrants = pd.Series('Q2', index=price_data.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Apply 1-day lag - we only know quadrant AFTER market close
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            
            # Get all periods where lagged quadrant was the current quadrant
            current_quad_mask = lagged_quadrants == current_quadrant
            current_quad_dates = lagged_quadrants[current_quad_mask].index
            
            if len(current_quad_dates) > 0:
                st.info(f"Analyzing asset performance during {len(current_quad_dates)} days in {current_quadrant} regime (with 1-day lag applied)")
                
                # Calculate performance for each asset during current quadrant periods
                asset_performance = []
                
                for symbol in analyzer.core_assets.keys():
                    if symbol in price_data.columns:
                        asset_data = price_data[symbol]
                        
                        # Get returns during current quadrant periods
                        quad_periods_returns = []
                        
                        for date in current_quad_dates:
                            if date in asset_data.index:
                                # Get same-day return (since we're already using lagged signals)
                                current_price = asset_data.loc[date]
                                
                                # For same-day return, we need previous day's price
                                prev_date_idx = asset_data.index.get_loc(date) - 1
                                if prev_date_idx >= 0:
                                    prev_price = asset_data.iloc[prev_date_idx]
                                    if pd.notna(current_price) and pd.notna(prev_price) and prev_price > 0:
                                        daily_return = (current_price / prev_price - 1) * 100
                                        quad_periods_returns.append(daily_return)
                        
                        if len(quad_periods_returns) >= 5:  # Need at least 5 observations
                            total_return = sum(quad_periods_returns)
                            avg_daily_return = np.mean(quad_periods_returns)
                            volatility = np.std(quad_periods_returns)
                            win_rate = (np.array(quad_periods_returns) > 0).mean() * 100
                            
                            # Calculate Sharpe-like ratio
                            sharpe = avg_daily_return / volatility if volatility > 0 else 0
                            
                            # Get recent performance (last 7 days)
                            recent_return = ((asset_data.iloc[-1] / asset_data.iloc[-8] - 1) * 100) if len(asset_data) >= 8 else 0
                            
                            asset_performance.append({
                                'Symbol': symbol,
                                'Name': analyzer.core_assets[symbol],
                                'Classification': analyzer.asset_classifications[symbol].primary_quadrant if symbol in analyzer.asset_classifications else 'Unknown',
                                'Total_Return_in_Quad': total_return,
                                'Avg_Daily_Return': avg_daily_return,
                                'Volatility': volatility,
                                'Sharpe_Ratio': sharpe,
                                'Win_Rate': win_rate,
                                'Days_Analyzed': len(quad_periods_returns),
                                'Recent_7d_Return': recent_return,
                                'Current_Price': asset_data.iloc[-1] if len(asset_data) > 0 else 0
                            })
                
                if asset_performance:
                    # Create DataFrame and sort by total return in current quadrant
                    assets_df = pd.DataFrame(asset_performance)
                    assets_df = assets_df.sort_values('Total_Return_in_Quad', ascending=False)
                    
                    # Asset ranking table
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"Asset Performance Ranking in {current_quadrant}")
                        
                        # Display table with performance metrics
                        display_df = assets_df[['Symbol', 'Name', 'Classification', 'Total_Return_in_Quad', 
                                              'Avg_Daily_Return', 'Win_Rate', 'Recent_7d_Return']].copy()
                        
                        # Format columns
                        display_df['Total_Return_in_Quad'] = display_df['Total_Return_in_Quad'].round(2)
                        display_df['Avg_Daily_Return'] = display_df['Avg_Daily_Return'].round(3)
                        display_df['Win_Rate'] = display_df['Win_Rate'].round(1)
                        display_df['Recent_7d_Return'] = display_df['Recent_7d_Return'].round(2)
                        
                        # Rename columns for display
                        display_df.columns = ['Symbol', 'Asset Name', 'Primary Quad', f'Total Return in {current_quadrant} (%)', 
                                            'Avg Daily Return (%)', 'Win Rate (%)', 'Recent 7d (%)']
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
                    
                    with col2:
                        st.subheader("Top/Bottom Performers")
                        
                        # Top 3 performers
                        st.markdown("**🏆 Top 3 Performers:**")
                        for i, (_, asset) in enumerate(assets_df.head(3).iterrows()):
                            st.markdown(f"{i+1}. **{asset['Symbol']}**: +{asset['Total_Return_in_Quad']:.1f}%")
                        
                        st.markdown("**📉 Bottom 3 Performers:**")
                        for i, (_, asset) in enumerate(assets_df.tail(3).iterrows()):
                            st.markdown(f"{i+1}. **{asset['Symbol']}**: {asset['Total_Return_in_Quad']:.1f}%")
                        
                        # Quadrant analysis
                        st.markdown("**📊 By Asset Class:**")
                        quad_performance = assets_df.groupby('Classification')['Total_Return_in_Quad'].mean().round(2)
                        for quad, perf in quad_performance.items():
                            st.markdown(f"• **{quad}** assets: {perf:+.1f}%")
                    
                    # Dynamic asset chart
                    st.subheader(f"Individual Asset Performance in {current_quadrant}")
                    
                    # Asset selector
                    selected_asset = st.selectbox(
                        "Select Asset to Analyze:",
                        options=assets_df['Symbol'].tolist(),
                        format_func=lambda x: f"{x} - {assets_df[assets_df['Symbol']==x]['Name'].iloc[0]}"
                    )
                    
                    if selected_asset and PLOTLY_AVAILABLE:
                        # Get selected asset data
                        selected_data = assets_df[assets_df['Symbol'] == selected_asset].iloc[0]
                        asset_price_data = price_data[selected_asset].copy()
                        
                        # Create performance chart for selected asset
                        fig = go.Figure()
                        
                        # Add full price line
                        fig.add_trace(go.Scatter(
                            x=asset_price_data.index,
                            y=asset_price_data.values,
                            mode='lines',
                            line=dict(color='lightgray', width=1),
                            name=f'{selected_asset} Price',
                            opacity=0.5
                        ))
                        
                        # Highlight current quadrant periods
                        for date in current_quad_dates:
                            if date in asset_price_data.index:
                                # Find start and end of consecutive quadrant periods
                                date_idx = asset_price_data.index.get_loc(date)
                                start_date = date
                                end_date = date
                                
                                # Extend to show periods
                                if date_idx > 0:
                                    start_idx = max(0, date_idx - 1)
                                    start_date = asset_price_data.index[start_idx]
                                
                                if date_idx < len(asset_price_data) - 1:
                                    end_idx = min(len(asset_price_data) - 1, date_idx + 1)
                                    end_date = asset_price_data.index[end_idx]
                                
                                # Add highlighted segment
                                segment_data = asset_price_data.loc[start_date:end_date]
                                
                                fig.add_trace(go.Scatter(
                                    x=segment_data.index,
                                    y=segment_data.values,
                                    mode='lines',
                                    line=dict(color='red', width=3),
                                    name=f'{current_quadrant} Period',
                                    showlegend=False
                                ))
                        
                        fig.update_layout(
                            title=f"{selected_asset} Performance During {current_quadrant} Periods",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show selected asset metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return in Quad", f"{selected_data['Total_Return_in_Quad']:.2f}%")
                        with col2:
                            st.metric("Win Rate", f"{selected_data['Win_Rate']:.1f}%")
                        with col3:
                            st.metric("Avg Daily Return", f"{selected_data['Avg_Daily_Return']:.3f}%")
                        with col4:
                            st.metric("Recent 7d Return", f"{selected_data['Recent_7d_Return']:.2f}%")
                    
                    elif selected_asset:
                        st.info("Install plotly for interactive asset charts")
                
                else:
                    st.warning("No assets have sufficient data for current quadrant analysis")
            else:
                st.warning("No periods found for current quadrant analysis")
        else:
            st.error("Failed to load quadrant analysis data")

    elif page == "Strategy Performance":
        st.markdown('<h1 class="main-header">Strategy Performance Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("Strategy Performance Analysis requires yfinance.")
            return
        
        # Load data
        with st.spinner("Loading strategy performance data..."):
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
        if daily_results is not None and price_data is not None:
            # Strategy selection
            strategy_type = st.selectbox(
                "Select Strategy Type:",
                ["Single Asset (BTC)", "Single Asset (ETH)", "Portfolio (50/50 BTC+ETH)"]
            )
            
            # Initialize strategy performance analyzer
            strategy_analyzer = StrategyPerformanceAnalysis()
            
            # Calculate performance based on selection
            if strategy_type == "Single Asset (BTC)":
                performance_data = strategy_analyzer.calculate_strategy_performance(
                    price_data, daily_results, 'BTC-USD', is_portfolio=False
                )
            elif strategy_type == "Single Asset (ETH)":
                performance_data = strategy_analyzer.calculate_strategy_performance(
                    price_data, daily_results, 'ETH-USD', is_portfolio=False
                )
            else:  # Portfolio
                performance_data = strategy_analyzer.calculate_strategy_performance(
                    price_data, daily_results, is_portfolio=True
                )
            
            if performance_data:
                # Display key metrics
                st.subheader("Performance Summary")
                
                strategy_metrics = performance_data['strategy_metrics']
                buyhold_metrics = performance_data['buyhold_metrics']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Strategy Total Return",
                        f"{strategy_metrics['total_return']:.1f}%",
                        delta=f"{strategy_metrics['total_return'] - buyhold_metrics['total_return']:+.1f}% vs B&H"
                    )
                
                with col2:
                    st.metric(
                        "Strategy Sharpe Ratio",
                        f"{strategy_metrics['sharpe_ratio']:.2f}",
                        delta=f"{strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']:+.2f} vs B&H"
                    )
                
                with col3:
                    st.metric(
                        "Max Drawdown",
                        f"{strategy_metrics['max_drawdown']:.1f}%",
                        delta=f"{strategy_metrics['max_drawdown'] - buyhold_metrics['max_drawdown']:+.1f}% vs B&H"
                    )
                
                with col4:
                    st.metric(
                        "Time in Market",
                        f"{performance_data['time_in_market']:.1f}%"
                    )
                
                with col5:
                    st.metric(
                        "Win Rate",
                        f"{strategy_metrics['win_rate']:.1f}%",
                        delta=f"{strategy_metrics['win_rate'] - buyhold_metrics['win_rate']:+.1f}% vs B&H"
                    )
                
                # Create performance charts
                strategy_analyzer.create_performance_charts(performance_data)
                
                # Detailed metrics table
                st.subheader("Detailed Performance Metrics")
                
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Total Return (%)',
                        'Annualized Return (%)',
                        'Volatility (%)',
                        'Sharpe Ratio',
                        'Max Drawdown (%)',
                        'Win Rate (%)'
                    ],
                    'Quadrant Strategy': [
                        f"{strategy_metrics['total_return']:.2f}",
                        f"{strategy_metrics['annualized_return']:.2f}",
                        f"{strategy_metrics['volatility']:.2f}",
                        f"{strategy_metrics['sharpe_ratio']:.2f}",
                        f"{strategy_metrics['max_drawdown']:.2f}",
                        f"{strategy_metrics['win_rate']:.2f}"
                    ],
                    'Buy & Hold': [
                        f"{buyhold_metrics['total_return']:.2f}",
                        f"{buyhold_metrics['annualized_return']:.2f}",
                        f"{buyhold_metrics['volatility']:.2f}",
                        f"{buyhold_metrics['sharpe_ratio']:.2f}",
                        f"{buyhold_metrics['max_drawdown']:.2f}",
                        f"{buyhold_metrics['win_rate']:.2f}"
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Strategy explanation
                st.subheader("Strategy Rules")
                st.info("""
                **Quadrant Strategy Rules:**
                1. **Long Position**: Enter when in Q1 (Goldilocks) or Q3 (Stagflation) quadrants AND price is above 50-day EMA
                2. **Flat Position**: All other conditions (Q2/Q4 quadrants OR price below 50-day EMA)
                3. **Signal Lag**: 1-day lag applied to avoid look-ahead bias
                4. **EMA Filter**: Must be above 50-day EMA to confirm trend strength
                
                **Performance Notes:**
                - Strategy aims to be long during favorable macro conditions with trend confirmation
                - EMA filter helps avoid false signals during sideways markets
                - Lower time in market can lead to better risk-adjusted returns
                """)
            
            else:
                st.error("Failed to calculate strategy performance")
        else:
            st.error("Failed to load data for strategy analysis")

    elif page == "Combined Dashboard":
        st.markdown('<h1 class="main-header">Crypto Macro Flow - Combined Dashboard</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("Combined Dashboard requires yfinance.")
            return
        
        # Load quadrant data
        with st.spinner("Loading comprehensive dashboard data..."):
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
        if daily_results is not None and price_data is not None:
            # Current status row
            current_quadrant = daily_results.iloc[-1]['Primary_Quadrant']
            current_score = daily_results.iloc[-1]['Primary_Score']
            
            st.subheader("🎯 Current Market Status")
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
                last_90_days = daily_results.tail(90)
                prev_score = last_90_days['Primary_Score'].iloc[-2] if len(last_90_days) > 1 else current_score
                st.metric("Quadrant Score", f"{current_score:.2f}", 
                         delta=f"{current_score - prev_score:.2f}")
            
            with col3:
                # BTC current price and change
                if 'BTC-USD' in price_data.columns:
                    btc_current = price_data['BTC-USD'].iloc[-1]
                    btc_prev = price_data['BTC-USD'].iloc[-2] if len(price_data) > 1 else btc_current
                    btc_change = ((btc_current / btc_prev - 1) * 100) if btc_prev > 0 else 0
                    st.metric("BTC Price", f"${btc_current:,.0f}", 
                             delta=f"{btc_change:+.2f}%")
            
            with col4:
                # ETH current price and change
                if 'ETH-USD' in price_data.columns:
                    eth_current = price_data['ETH-USD'].iloc[-1]
                    eth_prev = price_data['ETH-USD'].iloc[-2] if len(price_data) > 1 else eth_current
                    eth_change = ((eth_current / eth_prev - 1) * 100) if eth_prev > 0 else 0
                    st.metric("ETH Price", f"${eth_current:,.0f}", 
                             delta=f"{eth_change:+.2f}%")
            
            # Two-column layout for main content
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Quadrant Analysis")
                create_charts(price_data, daily_results, analyzer)
            
            with col2:
                st.subheader("💼 Strategy Performance")
                
                # Quick strategy performance
                strategy_analyzer = StrategyPerformanceAnalysis()
                performance_data = strategy_analyzer.calculate_strategy_performance(
                    price_data, daily_results, 'BTC-USD', is_portfolio=False
                )
                
                if performance_data:
                    strategy_metrics = performance_data['strategy_metrics']
                    buyhold_metrics = performance_data['buyhold_metrics']
                    
                    # Key performance metrics
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        st.metric(
                            "Strategy Return",
                            f"{strategy_metrics['total_return']:.1f}%",
                            delta=f"{strategy_metrics['total_return'] - buyhold_metrics['total_return']:+.1f}% vs B&H"
                        )
                        st.metric(
                            "Sharpe Ratio",
                            f"{strategy_metrics['sharpe_ratio']:.2f}",
                            delta=f"{strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']:+.2f}"
                        )
                    
                    with perf_col2:
                        st.metric(
                            "Max Drawdown",
                            f"{strategy_metrics['max_drawdown']:.1f}%",
                            delta=f"{strategy_metrics['max_drawdown'] - buyhold_metrics['max_drawdown']:+.1f}%"
                        )
                        st.metric(
                            "Time in Market",
                            f"{performance_data['time_in_market']:.1f}%"
                        )
                    
                    # Mini performance chart
                    if PLOTLY_AVAILABLE:
                        fig_mini = go.Figure()
                        
                        # Strategy performance
                        fig_mini.add_trace(go.Scatter(
                            x=strategy_metrics['cumulative_series'].index,
                            y=strategy_metrics['cumulative_series'].values * 100,
                            mode='lines',
                            name="Quadrant Strategy",
                            line=dict(color='#00ff00', width=2)
                        ))
                        
                        # Buy & Hold performance
                        fig_mini.add_trace(go.Scatter(
                            x=buyhold_metrics['cumulative_series'].index,
                            y=buyhold_metrics['cumulative_series'].values * 100,
                            mode='lines',
                            name="Buy & Hold",
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        fig_mini.update_layout(
                            title="Cumulative Returns",
                            xaxis_title="Date",
                            yaxis_title="Return (%)",
                            height=300,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        
                        st.plotly_chart(fig_mini, use_container_width=True)
            
            # Asset performance section with proper lag implementation
            st.subheader("🎯 Current Quadrant Asset Rankings")
            
            # APPLY SAME LAG LOGIC AS STRATEGY TO AVOID LOOK-AHEAD BIAS
            # Create lagged quadrant signals (shift by 1 day)
            aligned_quadrants = pd.Series('Q2', index=price_data.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Apply 1-day lag - we only know quadrant AFTER market close
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            
            # Get all periods where lagged quadrant was the current quadrant
            current_quad_mask = lagged_quadrants == current_quadrant
            current_quad_dates = lagged_quadrants[current_quad_mask].index
            
            if len(current_quad_dates) > 0:
                # Calculate performance for each asset during current quadrant periods
                asset_performance = []
                
                for symbol in analyzer.core_assets.keys():
                    if symbol in price_data.columns:
                        asset_data = price_data[symbol]
                        
                        # Get returns during current quadrant periods
                        quad_periods_returns = []
                        
                        for date in current_quad_dates:
                            if date in asset_data.index:
                                # Get same-day return (since we're already using lagged signals)
                                current_price = asset_data.loc[date]
                                
                                # For same-day return, we need previous day's price
                                prev_date_idx = asset_data.index.get_loc(date) - 1
                                if prev_date_idx >= 0:
                                    prev_price = asset_data.iloc[prev_date_idx]
                                    if pd.notna(current_price) and pd.notna(prev_price) and prev_price > 0:
                                        daily_return = (current_price / prev_price - 1) * 100
                                        quad_periods_returns.append(daily_return)
                        
                        if len(quad_periods_returns) >= 5:  # Need at least 5 observations
                            total_return = sum(quad_periods_returns)
                            avg_daily_return = np.mean(quad_periods_returns)
                            win_rate = (np.array(quad_periods_returns) > 0).mean() * 100
                            
                            # Get recent performance (last 7 days)
                            recent_return = ((asset_data.iloc[-1] / asset_data.iloc[-8] - 1) * 100) if len(asset_data) >= 8 else 0
                            
                            asset_performance.append({
                                'Symbol': symbol,
                                'Name': analyzer.core_assets[symbol],
                                'Classification': analyzer.asset_classifications[symbol].primary_quadrant if symbol in analyzer.asset_classifications else 'Unknown',
                                'Total_Return_in_Quad': total_return,
                                'Avg_Daily_Return': avg_daily_return,
                                'Win_Rate': win_rate,
                                'Recent_7d_Return': recent_return
                            })
                
                if asset_performance:
                    # Create DataFrame and sort by total return in current quadrant
                    assets_df = pd.DataFrame(asset_performance)
                    assets_df = assets_df.sort_values('Total_Return_in_Quad', ascending=False)
                    
                    # Show top 5 and bottom 5 performers
                    top_5 = assets_df.head(5)
                    bottom_5 = assets_df.tail(5)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🏆 Top 5 Performers in Current Quadrant:**")
                        top_display = top_5[['Symbol', 'Name', 'Total_Return_in_Quad', 'Recent_7d_Return']].copy()
                        top_display.columns = ['Symbol', 'Asset', f'{current_quadrant} Return (%)', 'Recent 7d (%)']
                        st.dataframe(top_display.round(2), hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.markdown("**📉 Bottom 5 Performers in Current Quadrant:**")
                        bottom_display = bottom_5[['Symbol', 'Name', 'Total_Return_in_Quad', 'Recent_7d_Return']].copy()
                        bottom_display.columns = ['Symbol', 'Asset', f'{current_quadrant} Return (%)', 'Recent 7d (%)']
                        st.dataframe(bottom_display.round(2), hide_index=True, use_container_width=True)
            
            # Crypto axe list section
            st.subheader("🚀 Crypto Axe List Generator")
            
            if st.button("Generate Crypto Axe List", type="primary"):
                try:
                    axe_generator = AxeListGenerator()
                    axe_list = axe_generator.run_analysis(top_n_tokens)
                    
                    if axe_list is not None and not axe_list.empty:
                        st.success(f"✅ Generated axe list with {len(axe_list)} tokens")
                        
                        # Show top 10 from axe list
                        top_10_axe = axe_list.head(10)
                        
                        display_axe_df = top_10_axe[['symbol', 'name', 'ratio_strength_score', 
                                                   'axe_score', 'month_return', 'week_return', 
                                                   'above_ma50', 'token_outperforming']].copy()
                        
                        display_axe_df.columns = ['Symbol', 'Name', 'Ratio Strength', 'Axe Score', 
                                                'Month %', 'Week %', 'Above MA50', 'Outperforming']
                        
                        st.dataframe(display_axe_df.round(2), hide_index=True, use_container_width=True)
                    else:
                        st.error("Failed to generate axe list")
                except Exception as e:
                    st.error(f"Error generating axe list: {e}")
            
            else:
                st.info("Click 'Generate Crypto Axe List' to analyze top crypto tokens for trading opportunities")
        
        else:
            st.error("Failed to load dashboard data")

if __name__ == "__main__":
    main()
