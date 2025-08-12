#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Updated with 90-day view and color-coded BTC chart
Live dashboard with quadrant analysis and axe list generator
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
            'api_delay': 1.0,            # Increased delay to avoid rate limits
            'max_retries': 2,            # Reduced retries to avoid hitting limits
            'default_top_n': 50,         # Reduced default to avoid rate limits
            'progress_interval': 5       # More frequent progress updates
        }
        if config:
            self.config.update(config)
    
    def get_top_tokens_by_market_cap(self, limit: int = 50) -> pd.DataFrame:
        # Use a smaller limit to avoid rate limits
        actual_limit = min(limit, 50)
        
        try:
            st.info(f"📊 Fetching top {actual_limit} tokens (reduced to avoid rate limits)...")
            
            url = f"{self.coingecko_url}/coins/markets"
            params = {
                'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': actual_limit,
                'page': 1, 'sparkline': False, 'locale': 'en'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            time.sleep(self.config['api_delay'])  # Longer delay
            
            data = response.json()
            if not data:
                st.error("No data returned from CoinGecko")
                return self._get_fallback_tokens()
            
            df = pd.DataFrame(data)
            df = df[['id', 'symbol', 'name', 'market_cap', 'market_cap_rank', 'current_price']].copy()
            df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
            df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')
            df['market_cap_rank'] = pd.to_numeric(df['market_cap_rank'], errors='coerce')
            df = df.dropna(subset=['market_cap', 'current_price', 'market_cap_rank'])
            df = df.sort_values('market_cap_rank').head(actual_limit)
            df['binance_symbol'] = df['symbol'].str.upper() + 'USDT'
            
            st.success(f"✅ Successfully fetched {len(df)} tokens from CoinGecko")
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.error("🚫 CoinGecko rate limit hit. Using fallback token list...")
                return self._get_fallback_tokens()
            else:
                st.error(f"❌ CoinGecko error: {e}")
                return self._get_fallback_tokens()
        except Exception as e:
            st.error(f"❌ Error fetching from CoinGecko: {e}")
            return self._get_fallback_tokens()
    
    def _get_fallback_tokens(self) -> pd.DataFrame:
        """Fallback token list when APIs fail"""
        st.info("📋 Using hardcoded top crypto tokens as fallback...")
        
        fallback_data = [
            {'id': 'bitcoin', 'symbol': 'btc', 'name': 'Bitcoin', 'market_cap': 2000000000000, 'market_cap_rank': 1, 'current_price': 100000},
            {'id': 'ethereum', 'symbol': 'eth', 'name': 'Ethereum', 'market_cap': 400000000000, 'market_cap_rank': 2, 'current_price': 4000},
            {'id': 'binancecoin', 'symbol': 'bnb', 'name': 'BNB', 'market_cap': 100000000000, 'market_cap_rank': 3, 'current_price': 600},
            {'id': 'solana', 'symbol': 'sol', 'name': 'Solana', 'market_cap': 80000000000, 'market_cap_rank': 4, 'current_price': 200},
            {'id': 'ripple', 'symbol': 'xrp', 'name': 'XRP', 'market_cap': 70000000000, 'market_cap_rank': 5, 'current_price': 1.2},
            {'id': 'cardano', 'symbol': 'ada', 'name': 'Cardano', 'market_cap': 20000000000, 'market_cap_rank': 6, 'current_price': 0.5},
            {'id': 'avalanche-2', 'symbol': 'avax', 'name': 'Avalanche', 'market_cap': 15000000000, 'market_cap_rank': 7, 'current_price': 40},
            {'id': 'polkadot', 'symbol': 'dot', 'name': 'Polkadot', 'market_cap': 12000000000, 'market_cap_rank': 8, 'current_price': 8},
            {'id': 'chainlink', 'symbol': 'link', 'name': 'Chainlink', 'market_cap': 11000000000, 'market_cap_rank': 9, 'current_price': 20},
            {'id': 'matic-network', 'symbol': 'matic', 'name': 'Polygon', 'market_cap': 10000000000, 'market_cap_rank': 10, 'current_price': 1.1},
            {'id': 'uniswap', 'symbol': 'uni', 'name': 'Uniswap', 'market_cap': 9000000000, 'market_cap_rank': 11, 'current_price': 12},
            {'id': 'litecoin', 'symbol': 'ltc', 'name': 'Litecoin', 'market_cap': 8000000000, 'market_cap_rank': 12, 'current_price': 100},
            {'id': 'near', 'symbol': 'near', 'name': 'NEAR Protocol', 'market_cap': 7000000000, 'market_cap_rank': 13, 'current_price': 7},
            {'id': 'algorand', 'symbol': 'algo', 'name': 'Algorand', 'market_cap': 6000000000, 'market_cap_rank': 14, 'current_price': 0.8},
            {'id': 'cosmos', 'symbol': 'atom', 'name': 'Cosmos Hub', 'market_cap': 5000000000, 'market_cap_rank': 15, 'current_price': 15}
        ]
        
        df = pd.DataFrame(fallback_data)
        df['binance_symbol'] = df['symbol'].str.upper() + 'USDT'
        
        return df
    
    def validate_binance_symbols(self, tokens_df: pd.DataFrame) -> pd.DataFrame:
        try:
            url = f"{self.binance_url}/exchangeInfo"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.config['api_delay'])
            
            data = response.json()
            available_symbols = set()
            
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['contractType'] == 'PERPETUAL' and
                    symbol_info['quoteAsset'] == 'USDT'):
                    available_symbols.add(symbol_info['symbol'])
            
            valid_tokens = []
            for _, token in tokens_df.iterrows():
                binance_symbol = token['binance_symbol']
                if binance_symbol in available_symbols:
                    valid_tokens.append(token)
            
            return pd.DataFrame(valid_tokens)
            
        except Exception as e:
            st.warning(f"⚠️ Binance validation failed ({e}), using all tokens")
            # Fallback: assume all common tokens are valid
            return tokens_df
    
    def get_coin_data(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        try:
            # Try Binance first
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            url = f"{self.binance_url}/klines"
            params = {
                'symbol': symbol, 'interval': '1d', 'startTime': start_time,
                'endTime': end_time, 'limit': 1000
            }
            
            response = self.session.get(url, params=params, timeout=10)
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
            
            if len(df) < days * 0.8:
                return None
            
            df['returns'] = df['price'].pct_change()
            df['ma_50'] = df['price'].rolling(window=50).mean()
            df['ma_20'] = df['price'].rolling(window=20).mean()
            df['above_ma50'] = df['price'] > df['ma_50']
            df['above_ma20'] = df['price'] > df['ma_20']
            df['ma50_distance'] = (df['price'] - df['ma_50']) / df['ma_50'] * 100
            
            return df
            
        except Exception as e:
            # Fallback: try CoinGecko for major coins
            try:
                coin_id = symbol.replace('USDT', '').lower()
                # Map common symbols to CoinGecko IDs
                coin_map = {
                    'btc': 'bitcoin', 'eth': 'ethereum', 'bnb': 'binancecoin',
                    'ada': 'cardano', 'xrp': 'ripple', 'sol': 'solana',
                    'dot': 'polkadot', 'doge': 'dogecoin', 'avax': 'avalanche-2',
                    'matic': 'matic-network', 'link': 'chainlink', 'uni': 'uniswap'
                }
                
                if coin_id in coin_map:
                    coin_id = coin_map[coin_id]
                
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {'vs_currency': 'usd', 'days': days}
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                time.sleep(0.5)  # Longer delay for CoinGecko
                
                data = response.json()
                prices = data.get('prices', [])
                
                if not prices:
                    return None
                
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['returns'] = df['price'].pct_change()
                df['ma_50'] = df['price'].rolling(window=50).mean()
                df['ma_20'] = df['price'].rolling(window=20).mean()
                df['above_ma50'] = df['price'] > df['ma_50']
                df['above_ma20'] = df['price'] > df['ma_20']
                df['ma50_distance'] = (df['price'] - df['ma_50']) / df['ma_50'] * 100
                
                return df
                
            except Exception:
                return None
    
    def determine_baseline_asset(self) -> str:
        try:
            st.info("🔍 Analyzing BTC/ETH pair to determine baseline...")
            
            btc_data = self.get_coin_data('BTCUSDT', days=100)
            time.sleep(0.3)
            eth_data = self.get_coin_data('ETHUSDT', days=100)
            
            if btc_data is None or eth_data is None:
                st.warning("⚠️ Could not fetch BTC/ETH data, defaulting to BTC baseline")
                return 'BTC'
            
            btceth_ratio = btc_data['price'] / eth_data['price']
            ratio_ma_50 = btceth_ratio.rolling(window=50).mean()
            current_ratio = btceth_ratio.iloc[-1]
            current_ma_50 = ratio_ma_50.iloc[-1]
            
            btc_outperforming = current_ratio > current_ma_50
            
            if btc_outperforming:
                st.success("✅ Baseline determined: **BTC** (outperforming ETH)")
                return 'BTC'
            else:
                st.success("✅ Baseline determined: **ETH** (outperforming BTC)")
                return 'ETH'
                
        except Exception as e:
            st.error(f"Error determining baseline: {e}")
            return 'BTC'
    
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
    
    def calculate_ratio_ma_ranking(self, token_symbol: str, baseline_symbol: str) -> Optional[Dict]:
        try:
            token_data = self.get_coin_data(token_symbol, days=100)
            if token_data is None:
                return None
            
            baseline_data = self.get_coin_data(baseline_symbol, days=100)
            if baseline_data is None:
                return None
            
            if len(token_data) < 80 or len(baseline_data) < 80:
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
            
            # Calculate correlation and beta
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
            
            analysis = {
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
            
            return analysis
            
        except Exception:
            return None
    
    def generate_axe_list(self, top_tokens: pd.DataFrame, baseline_asset: str) -> pd.DataFrame:
        baseline_symbol = 'BTCUSDT' if baseline_asset == 'BTC' else 'ETHUSDT'
        analysis_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (idx, token) in enumerate(top_tokens.iterrows()):
            binance_symbol = token['binance_symbol']
            name = token['name']
            
            if binance_symbol in ['BTCUSDT', 'ETHUSDT']:
                continue
            
            progress = (i + 1) / len(top_tokens)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {name} ({i+1}/{len(top_tokens)})...")
            
            analysis = self.analyze_token_performance(binance_symbol, baseline_asset)
            if analysis:
                ratio_analysis = self.calculate_ratio_ma_ranking(binance_symbol, baseline_symbol)
                
                if ratio_analysis:
                    analysis.update(ratio_analysis)
