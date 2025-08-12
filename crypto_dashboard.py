#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Fixed Version with Better Error Handling
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
import json

# Handle imports with proper error checking
YFINANCE_AVAILABLE = True
PLOTLY_AVAILABLE = True

try:
    import yfinance as yf
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not available - install with: pip install yfinance")

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ plotly not available - install with: pip install plotly")

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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(self.core_assets.keys()):
            progress = (i + 1) / len(self.core_assets)
            progress_bar.progress(progress)
            status_text.text(f"Fetching {symbol}...")
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, timeout=10)
                if len(hist) > 0:
                    clean_close = hist['Close'].dropna()
                    clean_close = clean_close[clean_close > 0]
                    if len(clean_close) >= 10:
                        data[symbol] = clean_close
                time.sleep(0.1)  # Small delay to avoid rate limiting
            except Exception as e:
                st.warning(f"⚠️ Failed to fetch {symbol}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not data:
            st.error("❌ No data fetched successfully")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.groupby(df.index.date).last()
            df.index = pd.to_datetime(df.index)
            df = df.fillna(method='ffill').dropna(how='all')
            st.success(f"✅ Successfully fetched data for {len(df.columns)} assets")
        
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
# IMPROVED AXE LIST GENERATOR MODULE
# ================================================================================================

class ImprovedAxeListGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        })
        
        self.config = {
            'api_delay': 1.0,  # Increased delay
            'max_retries': 3,
            'retry_backoff_base': 2,
            'default_top_n': 50,  # Reduced default
            'progress_interval': 10,
            'timeout': 30
        }
        if config:
            self.config.update(config)
    
    def get_top_tokens_by_market_cap(self, limit: int = 50, max_retries: Optional[int] = None) -> pd.DataFrame:
        """Fetch top tokens from CoinGecko with improved error handling"""
        if max_retries is None:
            max_retries = self.config['max_retries']
        
        st.info(f"🔍 Fetching top {limit} tokens from CoinGecko...")
        
        for attempt in range(max_retries):
            try:
                url = f"{self.coingecko_url}/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': min(limit, 250),  # CoinGecko limit
                    'page': 1,
                    'sparkline': False,
                    'locale': 'en',
                    'price_change_percentage': '7d,30d'
                }
                
                response = self.session.get(url, params=params, timeout=self.config['timeout'])
                response.raise_for_status()
                
                data = response.json()
                if not data:
                    st.error("❌ No data returned from CoinGecko")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                required_cols = ['id', 'symbol', 'name', 'market_cap', 'market_cap_rank', 'current_price']
                available_cols = [col for col in required_cols if col in df.columns]
                
                if len(available_cols) < 5:
                    st.error(f"❌ Missing required columns. Available: {available_cols}")
                    return pd.DataFrame()
                
                df = df[available_cols].copy()
                
                # Add price change data if available
                if 'price_change_percentage_7d_in_currency' in df.columns:
                    df['week_return'] = pd.to_numeric(df['price_change_percentage_7d_in_currency'], errors='coerce')
                if 'price_change_percentage_30d_in_currency' in df.columns:
                    df['month_return'] = pd.to_numeric(df['price_change_percentage_30d_in_currency'], errors='coerce')
                
                # Clean and convert data
                for col in ['market_cap', 'current_price', 'market_cap_rank']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['market_cap', 'current_price', 'market_cap_rank'])
                df = df.sort_values('market_cap_rank').head(limit)
                
                st.success(f"✅ Successfully fetched {len(df)} tokens from CoinGecko")
                time.sleep(self.config['api_delay'])
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (self.config['retry_backoff_base'] ** attempt) * 10
                    st.warning(f"⚠️ Rate limited by CoinGecko. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"❌ HTTP error {e.response.status_code}: {e}")
                    break
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {e}")
                break
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                break
        
        return pd.DataFrame()
    
    def get_crypto_data_yfinance(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """Fallback method using yfinance for crypto data"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # Convert symbol to Yahoo Finance format
            if symbol.upper().endswith('USDT'):
                yf_symbol = symbol.upper().replace('USDT', '-USD')
            else:
                yf_symbol = f"{symbol.upper()}-USD"
            
            ticker = yf.Ticker(yf_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)
            
            hist = ticker.history(start=start_date, end=end_date, timeout=10)
            
            if hist.empty or len(hist) < days // 2:
                return None
            
            df = hist.copy()
            df['price'] = df['Close']
            df['returns'] = df['price'].pct_change()
            df['ma_50'] = df['price'].rolling(window=min(50, len(df)//2)).mean()
            df['ma_20'] = df['price'].rolling(window=min(20, len(df)//4)).mean()
            df['above_ma50'] = df['price'] > df['ma_50']
            df['above_ma20'] = df['price'] > df['ma_20']
            df['ma50_distance'] = (df['price'] - df['ma_50']) / df['ma_50'] * 100
            
            return df
            
        except Exception:
            return None
    
    def calculate_simple_metrics(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Calculate performance metrics from price data"""
        if df is None or len(df) < 20:
            return None
        
        try:
            latest = df.iloc[-1]
            week_ago = df.iloc[-min(8, len(df)-1)]
            month_ago = df.iloc[-min(31, len(df)-1)]
            
            if pd.isna(latest['price']):
                return None
            
            # Calculate returns
            week_return = (latest['price'] / week_ago['price'] - 1) * 100 if not pd.isna(week_ago['price']) else 0
            month_return = (latest['price'] / month_ago['price'] - 1) * 100 if not pd.isna(month_ago['price']) else 0
            
            # Calculate volatility
            returns = df['returns'].dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            
            # MA analysis
            above_ma50 = latest['above_ma50'] if not pd.isna(latest['above_ma50']) else False
            above_ma20 = latest['above_ma20'] if not pd.isna(latest['above_ma20']) else False
            ma50_distance = latest['ma50_distance'] if not pd.isna(latest['ma50_distance']) else 0
            
            return {
                'symbol': symbol,
                'current_price': latest['price'],
                'above_ma50': above_ma50,
                'above_ma20': above_ma20,
                'ma50_distance': ma50_distance,
                'week_return': week_return,
                'month_return': month_return,
                'volatility': volatility,
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0
            }
            
        except Exception:
            return None
    
    def analyze_token_simple(self, symbol: str, baseline_return: float = 0) -> Optional[Dict]:
        """Simplified token analysis using yfinance"""
        try:
            data = self.get_crypto_data_yfinance(symbol, days=100)
            if data is None:
                return None
            
            metrics = self.calculate_simple_metrics(data, symbol)
            if metrics is None:
                return None
            
            # Calculate relative strength vs baseline
            metrics['relative_strength'] = metrics['month_return'] - baseline_return
            metrics['token_outperforming'] = metrics['relative_strength'] > 0
            
            # Simple scoring
            metrics['axe_score'] = (
                int(metrics['above_ma50']) * 2 +
                int(metrics['above_ma20']) * 1 +
                int(metrics['ma50_distance'] > 0) * 1 +
                int(metrics['week_return'] > 0) * 1 +
                int(metrics['month_return'] > 0) * 1 +
                int(metrics['relative_strength'] > 0) * 2
            )
            
            return metrics
            
        except Exception:
            return None
    
    def run_simplified_analysis(self, top_n: int = 30):
        """Run a simplified analysis that's more reliable"""
        st.info(f"🚀 Starting simplified analysis for top {top_n} tokens...")
        
        # Get top tokens
        top_tokens = self.get_top_tokens_by_market_cap(top_n)
        if top_tokens.empty:
            st.error("❌ Failed to fetch token data")
            return None
        
        # Get baseline performance (BTC)
        st.info("📊 Analyzing BTC baseline performance...")
        btc_data = self.get_crypto_data_yfinance('BTC', days=100)
        baseline_return = 0
        
        if btc_data is not None:
            baseline_metrics = self.calculate_simple_metrics(btc_data, 'BTC')
            if baseline_metrics:
                baseline_return = baseline_metrics['month_return']
                st.success(f"✅ BTC baseline monthly return: {baseline_return:.1f}%")
        
        # Analyze tokens
        st.info(f"🔍 Analyzing {len(top_tokens)} tokens...")
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (_, token) in enumerate(top_tokens.iterrows()):
            progress = (i + 1) / len(top_tokens)
            progress_bar.progress(progress)
            status_text.text(f"[{i+1}/{len(top_tokens)}] Analyzing {token['symbol'].upper()}...")
            
            if token['symbol'].upper() in ['BTC', 'BITCOIN']:
                continue
            
            analysis = self.analyze_token_simple(token['symbol'], baseline_return)
            
            if analysis:
                # Add token metadata
                analysis.update({
                    'name': token['name'],
                    'market_cap_rank': token['market_cap_rank'],
                    'market_cap': token['market_cap']
                })
                
                # Use existing return data if available
                if 'week_return' in token and not pd.isna(token['week_return']):
                    analysis['week_return'] = token['week_return']
                if 'month_return' in token and not pd.isna(token['month_return']):
                    analysis['month_return'] = token['month_return']
                    analysis['relative_strength'] = token['month_return'] - baseline_return
                    analysis['token_outperforming'] = analysis['relative_strength'] > 0
                
                results.append(analysis)
            
            time.sleep(0.5)  # Longer delay for stability
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values(['relative_strength', 'axe_score'], ascending=[False, False])
            st.success(f"🎯 Analysis complete! Successfully analyzed {len(df)} tokens")
            return df
        else:
            st.error("❌ No successful analyses")
            return None

# ================================================================================================
# STREAMLIT DASHBOARD
# ================================================================================================

def main():
    st.set_page_config(
        page_title="Crypto Macro Flow Dashboard - Fixed",
        page_icon="📊", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    st.title("📊 Crypto Macro Flow Dashboard - Fixed Version")
    
    # Display status
    col1, col2 = st.columns(2)
    with col1:
        if YFINANCE_AVAILABLE:
            st.success("✅ yfinance: Available")
        else:
            st.error("❌ yfinance: Not available")
    
    with col2:
        if PLOTLY_AVAILABLE:
            st.success("✅ plotly: Available")
        else:
            st.warning("⚠️ plotly: Not available")

    # Sidebar
    st.sidebar.title("🎯 Dashboard Controls")

    # Navigation
    page = st.sidebar.selectbox("📊 Select Analysis", 
        ["Current Quadrant Analysis", "Simplified Axe List", "Combined Dashboard"])

    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    lookback_days = st.sidebar.slider("Momentum Lookback (days)", 14, 50, 21)
    top_n_tokens = st.sidebar.slider("Top N Tokens for Analysis", 10, 50, 20)

    # Main content based on page selection
    if page == "Current Quadrant Analysis":
        st.header("📊 Current Quadrant Analysis")
        
        if not YFINANCE_AVAILABLE:
            st.error("❌ This feature requires yfinance. Install with: `pip install yfinance`")
            return
        
        if st.button("🔄 Run Quadrant Analysis", type="primary"):
            with st.spinner("Running quadrant analysis..."):
                try:
                    analyzer = CurrentQuadrantAnalysis(lookback_days=lookback_days)
                    price_data = analyzer.fetch_recent_data(days_back=365)
                    
                    if not price_data.empty:
                        momentum_data = analyzer.calculate_daily_momentum(price_data)
                        quadrant_scores = analyzer.calculate_daily_quadrant_scores(momentum_data)
                        daily_results = analyzer.determine_daily_quadrant(quadrant_scores)
                        
                        # Store in session state
                        st.session_state['quadrant_data'] = {
                            'price_data': price_data,
                            'daily_results': daily_results,
                            'analyzer': analyzer
                        }
                        
                        st.success("✅ Quadrant analysis complete!")
                    else:
                        st.error("❌ Failed to fetch price data")
                        
                except Exception as e:
                    st.error(f"❌ Error in quadrant analysis: {str(e)}")
        
        # Display results if available
        if 'quadrant_data' in st.session_state:
            data = st.session_state['quadrant_data']
            price_data = data['price_data']
            daily_results = data['daily_results']
            analyzer = data['analyzer']
            
            # Current status
            if not daily_results.empty:
                current_data = daily_results.tail(1).iloc[0]
                current_quadrant = current_data['Primary_Quadrant']
                current_score = current_data['Primary_Score']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Current Quadrant", current_quadrant)
                    st.write(analyzer.quadrant_descriptions[current_quadrant])
                
                with col2:
                    st.metric("Primary Score", f"{current_score:.2f}")
                
                with col3:
                    confidence_val = current_data['Confidence']
                    confidence_text = "Very High" if np.isinf(confidence_val) else f"{confidence_val:.2f}"
                    st.metric("Confidence", confidence_text)
                
                # Charts
                if 'BTC-USD' in price_data.columns:
                    st.subheader("📈 Bitcoin Price (Last Year)")
                    st.line_chart(price_data['BTC-USD'].tail(365))
                
                # Recent quadrant scores
                st.subheader("📊 Recent Quadrant Scores")
                recent_scores = daily_results.tail(30)[['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score']]
                st.line_chart(recent_scores)

    elif page == "Simplified Axe List":
        st.header("🎯 Simplified Axe List Generator")
        st.info("This version uses a simplified approach with better error handling and yfinance data.")
        
        if st.button("🚀 Generate Simplified Axe List", type="primary"):
            try:
                generator = ImprovedAxeListGenerator()
                results = generator.run_simplified_analysis(top_n_tokens)
                
                if results is not None and not results.empty:
                    st.session_state['axe_data'] = results
                    st.success(f"✅ Successfully analyzed {len(results)} tokens!")
                else:
                    st.error("❌ Failed to generate axe list")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Display results
        if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
            axe_data = st.session_state['axe_data']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏆 Top Performer", axe_data.iloc[0]['name'][:15])
            
            with col2:
                best_return = axe_data['month_return'].max()
                st.metric("Best Monthly Return", f"{best_return:.1f}%")
            
            with col3:
                outperforming = axe_data['token_outperforming'].sum()
                st.metric("Outperforming BTC", f"{outperforming}/{len(axe_data)}")
            
            with col4:
                avg_score = axe_data['axe_score'].mean()
                st.metric("Avg Axe Score", f"{avg_score:.1f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 10: Monthly Returns")
                top_10_returns = axe_data.head(10).set_index('symbol')['month_return']
                st.bar_chart(top_10_returns)
            
            with col2:
                st.subheader("📊 Week vs Month Returns")
                scatter_data = axe_data[['week_return', 'month_return']].head(15)
                st.scatter_chart(scatter_data)
            
            # Detailed table
            st.subheader("📋 Detailed Results")
            display_cols = ['name', 'symbol', 'month_return', 'week_return', 'above_ma50', 'above_ma20', 'axe_score', 'market_cap_rank']
            available_cols = [col for col in display_cols if col in axe_data.columns]
            
            display_df = axe_data[available_cols].copy()
            if 'market_cap_rank' in display_df.columns:
                display_df['market_cap_rank'] = display_df['market_cap_rank'].astype(int)
            
            st.dataframe(display_df.round(2), use_container_width=True)

    else:  # Combined Dashboard
        st.header("🚀 Combined Dashboard")
        st.info("Run both analyses for a complete overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Run Quadrant Analysis"):
                if YFINANCE_AVAILABLE:
                    with st.spinner("Running quadrant analysis..."):
                        try:
                            analyzer = CurrentQuadrantAnalysis(lookback_days=lookback_days)
                            price_data = analyzer.fetch_recent_data(days_back=365)
                            
                            if not price_data.empty:
                                momentum_data = analyzer.calculate_daily_momentum(price_data)
                                quadrant_scores = analyzer.calculate_daily_quadrant_scores(momentum_data)
                                daily_results = analyzer.determine_daily_quadrant(quadrant_scores)
                                
                                st.session_state['quadrant_data'] = {
                                    'price_data': price_data,
                                    'daily_results': daily_results,
                                    'analyzer': analyzer
                                }
                                st.success("✅ Quadrant analysis complete!")
                            else:
                                st.error("❌ Failed to fetch price data")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.error("❌ yfinance required")
        
        with col2:
            if st.button("🎯 Run Axe Analysis"):
                with st.spinner("Running axe analysis..."):
                    try:
                        generator = ImprovedAxeListGenerator()
                        results = generator.run_simplified_analysis(top_n_tokens)
                        
                        if results is not None and not results.empty:
                            st.session_state['axe_data'] = results
                            st.success(f"✅ Axe analysis complete!")
                        else:
                            st.error("❌ Failed to generate axe list")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display combined results
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current Market Regime")
            if 'quadrant_data' in st.session_state:
                data = st.session_state['quadrant_data']
                daily_results = data['daily_results']
                analyzer = data['analyzer']
                
                if not daily_results.empty:
                    current_quadrant = daily_results.tail(1).iloc[0]['Primary_Quadrant']
                    st.metric("Current Quadrant", current_quadrant)
                    st.write(analyzer.quadrant_descriptions[current_quadrant])
                    
                    # Mini chart
                    if 'BTC-USD' in data['price_data'].columns:
                        st.line_chart(data['price_data']['BTC-USD'].tail(90))
            else:
                st.info("Run quadrant analysis to see current regime")
        
        with col2:
            st.subheader("🎯 Top Performing Tokens")
            if 'axe_data' in st.session_state and not st.session_state['axe_data'].empty:
                axe_data = st.session_state['axe_data']
                
                # Top 5 performers
                top_5 = axe_data.head(5)[['name', 'month_return', 'axe_score']]
                top_5.columns = ['Token', 'Monthly Return (%)', 'Axe Score']
                st.dataframe(top_5.round(2), use_container_width=True)
                
                # Performance metrics
                outperforming = axe_data['token_outperforming'].sum()
                st.metric("Tokens Outperforming BTC", f"{outperforming}/{len(axe_data)}")
            else:
                st.info("Run axe analysis to see top performers")

    # Footer with instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 What's Fixed")
    st.sidebar.markdown("""
    **✅ Improvements:**
    - Better error handling for API failures
    - Fallback to yfinance for crypto data
    - Reduced API calls to avoid rate limiting
    - More robust data validation
    - Simplified analysis approach
    - Improved progress indicators
    
    **🔧 Requirements:**
    - `pip install yfinance pandas numpy streamlit`
    - `pip install plotly` (optional, for better charts)
    
    **⚠️ Notes:**
    - Binance API issues resolved by using yfinance
    - Reduced token count for better reliability
    - CoinGecko rate limits handled gracefully
    """)
    
    st.sidebar.markdown("### 🆘 Troubleshooting")
    st.sidebar.markdown("""
    **If you still see errors:**
    1. Check internet connection
    2. Try reducing the number of tokens
    3. Wait a few minutes between runs
    4. Restart the Streamlit app
    """)

if __name__ == "__main__":
    main()
