#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Updated with 90-day view and color-coded BTC chart
Live dashboard with quadrant analysis and strategy performance analysis
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
# STRATEGY PERFORMANCE MODULE
# ================================================================================================

class StrategyPerformanceAnalysis:
    def __init__(self):
        self.strategy_name = "Quadrant Strategy"
        self.benchmark_name = "Buy & Hold"
    
    def calculate_strategy_performance(self, price_data: pd.DataFrame, daily_results: pd.DataFrame) -> Dict:
        """Calculate performance metrics for quadrant strategy vs buy & hold"""
        
        if price_data is None or 'BTC-USD' not in price_data.columns or daily_results is None:
            return None
        
        try:
            # Align price data with quadrant data
            btc_prices = price_data['BTC-USD'].copy()
            
            # Forward fill quadrant data to match price data length
            aligned_quadrants = pd.Series('Q2', index=btc_prices.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Forward fill quadrant assignments
            aligned_quadrants = aligned_quadrants.fillna(method='ffill')
            
            # CRITICAL FIX: Apply 1-day lag to avoid look-ahead bias
            # Shift quadrant signals forward by 1 day (lag the signal)
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            
            # Calculate daily returns
            btc_returns = btc_prices.pct_change().fillna(0)
            
            # Strategy: Long in Q1+Q3, Flat in Q2+Q4 (using lagged signals)
            strategy_positions = lagged_quadrants.isin(['Q1', 'Q3']).astype(int)
            strategy_returns = btc_returns * strategy_positions
            
            # Buy & Hold returns
            buyhold_returns = btc_returns
            
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
                'btc_prices': btc_prices,
                'signal_lag_applied': True  # Flag to indicate proper implementation
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
            name=self.strategy_name,
            line=dict(color='#00ff00', width=2)
        ))
        
        # Buy & Hold performance
        fig_performance.add_trace(go.Scatter(
            x=buyhold_metrics['cumulative_series'].index,
            y=buyhold_metrics['cumulative_series'].values * 100,
            mode='lines',
            name=self.benchmark_name,
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
        
        # Chart 2: Strategy Positions & BTC Price
        fig_positions = go.Figure()
        
        # BTC Price (secondary y-axis)
        fig_positions.add_trace(go.Scatter(
            x=performance_data['btc_prices'].index,
            y=performance_data['btc_prices'].values,
            mode='lines',
            name='BTC Price',
            line=dict(color='orange', width=1),
            yaxis='y2'
        ))
        
        # Strategy positions
        positions_for_plot = performance_data['strategy_positions'] * performance_data['btc_prices'].max() * 0.1
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
            title="Strategy Positions vs BTC Price",
            xaxis_title="Date",
            yaxis=dict(title="Position Signal", side="left"),
            yaxis2=dict(title="BTC Price (USD)", side="right", overlaying="y"),
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

    elif page == "Strategy Performance":
        st.markdown('<h1 class="main-header">Strategy Performance Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("Strategy Performance Analysis requires yfinance.")
            return
        
        # Load data
        with st.spinner("Loading strategy performance data..."):
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
        if daily_results is not None and price_data is not None:
            # Calculate strategy performance
            strategy_analyzer = StrategyPerformanceAnalysis()
            performance_data = strategy_analyzer.calculate_strategy_performance(price_data, daily_results)
            
            if performance_data:
                strategy_metrics = performance_data['strategy_metrics']
                buyhold_metrics = performance_data['buyhold_metrics']
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Strategy Total Return", 
                             f"{strategy_metrics['total_return']:.1f}%",
                             delta=f"{strategy_metrics['total_return'] - buyhold_metrics['total_return']:.1f}% vs B&H")
                
                with col2:
                    st.metric("Sharpe Ratio", 
                             f"{strategy_metrics['sharpe_ratio']:.2f}",
                             delta=f"{strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", 
                             f"{strategy_metrics['max_drawdown']:.1f}%",
                             delta=f"{strategy_metrics['max_drawdown'] - buyhold_metrics['max_drawdown']:.1f}%")
                
                with col4:
                    st.metric("Time in Market", 
                             f"{performance_data['time_in_market']:.1f}%")
                
                # Performance charts
                strategy_analyzer.create_performance_charts(performance_data)
                
                # Detailed metrics comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quadrant Strategy Metrics")
                    strategy_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                                  'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                        'Value': [f"{strategy_metrics['total_return']:.2f}",
                                 f"{strategy_metrics['annualized_return']:.2f}",
                                 f"{strategy_metrics['volatility']:.2f}",
                                 f"{strategy_metrics['sharpe_ratio']:.2f}",
                                 f"{strategy_metrics['max_drawdown']:.2f}",
                                 f"{strategy_metrics['win_rate']:.2f}"]
                    })
                    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("Buy & Hold Metrics")
                    buyhold_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                                  'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                        'Value': [f"{buyhold_metrics['total_return']:.2f}",
                                 f"{buyhold_metrics['annualized_return']:.2f}",
                                 f"{buyhold_metrics['volatility']:.2f}",
                                 f"{buyhold_metrics['sharpe_ratio']:.2f}",
                                 f"{buyhold_metrics['max_drawdown']:.2f}",
                                 f"{buyhold_metrics['win_rate']:.2f}"]
                    })
                    st.dataframe(buyhold_df, use_container_width=True, hide_index=True)
                
                # Strategy summary
                st.subheader("Strategy Summary")
                
                outperformance = strategy_metrics['total_return'] - buyhold_metrics['total_return']
                risk_adjusted = strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']
                
                if outperformance > 0:
                    perf_color = "GREEN"
                    perf_text = "outperformed"
                else:
                    perf_color = "RED"
                    perf_text = "underperformed"
                
                st.markdown(f"""
                **Strategy Analysis:**
                - {perf_color}: The Quadrant Strategy **{perf_text}** Buy & Hold by **{outperformance:+.1f}%**
                - **Risk-Adjusted Performance**: Sharpe ratio difference of **{risk_adjusted:+.2f}**
                - **Market Exposure**: Only **{performance_data['time_in_market']:.1f}%** of the time ({performance_data['long_days']} days long, {performance_data['flat_days']} days flat)
                - **Strategy Logic**: Long during Q1 (Goldilocks) and Q3 (Stagflation), flat during Q2 (Reflation) and Q4 (Deflation)
                """)
                
                # Technical implementation note
                st.info("""
                **Strategy Implementation Notes:**
                
                **Signal Lag Applied**: This backtest uses **1-day lagged signals** to avoid look-ahead bias:
                - **Day T-1**: Calculate quadrant using momentum data through T-1
                - **Day T**: Apply that signal to Day T's trading (realistic implementation)
                - **No Cheating**: We never use "today's" price to make "today's" trading decision
                
                **Real Trading**: In practice, you'd calculate the quadrant at market close and apply the position the next day.
                """)
                
            else:
                st.error("Failed to calculate strategy performance.")
        else:
            st.error("Failed to load strategy performance data.")

    else:  # Combined Dashboard
        st.markdown('<h1 class="main-header">Combined Macro Flow Dashboard</h1>', unsafe_allow_html=True)
        
        # Load quadrant data
        if YFINANCE_AVAILABLE:
            with st.spinner("Loading quadrant analysis..."):
                price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        else:
            price_data, daily_results, analyzer = None, None, None
        
        if daily_results is not None:
            # Current status row - updated to use 90 days
            current_data = daily_results.tail(90).iloc[-1]
            current_quadrant = current_data['Primary_Quadrant']
            
            # Calculate quick strategy performance
            strategy_analyzer = StrategyPerformanceAnalysis()
            performance_data = strategy_analyzer.calculate_strategy_performance(price_data, daily_results)
            
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
                if performance_data:
                    strategy_return = performance_data['strategy_metrics']['total_return']
                    buyhold_return = performance_data['buyhold_metrics']['total_return']
                    outperformance = strategy_return - buyhold_return
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Strategy Performance</h4>
                        <h3>{strategy_return:+.1f}%</h3>
                        <p>Outperformance: {outperformance:+.1f}%</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Strategy Performance</h4>
                        <h3>Loading...</h3>
                        <p>Calculating returns</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col3:
                if performance_data:
                    time_in_market = performance_data['time_in_market']
                    position = "LONG" if current_quadrant in ['Q1', 'Q3'] else "FLAT"
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Current Position</h4>
                        <h3>{position}</h3>
                        <p>Time in market: {time_in_market:.1f}%</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>Current Position</h4>
                        <h3>Loading...</h3>
                        <p>Analyzing position</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Main charts - updated to use 90 days
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if price_data is not None:
                    st.subheader("Bitcoin Price Trend - Last 90 Days")
                    st.line_chart(price_data['BTC-USD'].tail(90))
            
            with col2:
                if performance_data and PLOTLY_AVAILABLE:
                    st.subheader("Strategy vs Buy & Hold (Last 30 Days)")
                    last_30_strategy = performance_data['strategy_metrics']['cumulative_series'].tail(30)
                    last_30_buyhold = performance_data['buyhold_metrics']['cumulative_series'].tail(30)
                    
                    chart_data = pd.DataFrame({
                        'Strategy': (last_30_strategy.values - 1) * 100,
                        'Buy & Hold': (last_30_buyhold.values - 1) * 100
                    }, index=last_30_strategy.index)
                    
                    st.line_chart(chart_data)
                else:
                    st.info("Strategy performance chart loading...")
            
            # Quick stats - updated to use 90 days
            col1, col2 = st.columns(2)
            
            with col1:
                if daily_results is not None:
                    st.subheader("Recent Quadrant Trend (Last 7 Days)")
                    recent_quads = daily_results.tail(7)['Primary_Quadrant'].value_counts()
                    st.bar_chart(recent_quads)
            
            with col2:
                st.subheader("Strategy Key Stats")
                if performance_data:
                    key_stats = pd.DataFrame({
                        'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                        'Strategy': [f"{performance_data['strategy_metrics']['total_return']:.1f}%",
                                   f"{performance_data['strategy_metrics']['sharpe_ratio']:.2f}",
                                   f"{performance_data['strategy_metrics']['max_drawdown']:.1f}%",
                                   f"{performance_data['strategy_metrics']['win_rate']:.1f}%"],
                        'Buy & Hold': [f"{performance_data['buyhold_metrics']['total_return']:.1f}%",
                                     f"{performance_data['buyhold_metrics']['sharpe_ratio']:.2f}",
                                     f"{performance_data['buyhold_metrics']['max_drawdown']:.1f}%",
                                     f"{performance_data['buyhold_metrics']['win_rate']:.1f}%"]
                    })
                    st.dataframe(key_stats, use_container_width=True, hide_index=True)
                else:
                    st.info("Loading strategy statistics...")
            
        else:
            st.error("Failed to load dashboard data. Please refresh the page.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Sources")
    st.sidebar.markdown("• **Quadrant Analysis**: Yahoo Finance")
    st.sidebar.markdown("• **Strategy Performance**: Calculated returns")
    st.sidebar.markdown("• **Refresh Rate**: 5 minutes")
    
    # Status indicators
    st.sidebar.markdown("### System Status")
    if YFINANCE_AVAILABLE:
        st.sidebar.markdown("• Yahoo Finance: Ready")
    else:
        st.sidebar.markdown("• Yahoo Finance: Missing")
    
    if PLOTLY_AVAILABLE:
        st.sidebar.markdown("• Plotly Charts: Ready")
    else:
        st.sidebar.markdown("• Plotly Charts: Basic mode")
    
    # Instructions
    with st.sidebar.expander("How to Use"):
        st.markdown("""
        **Quadrant Analysis:**
        - Shows current market regime (Growth/Inflation)
        - BTC chart colored by quadrant periods
        - Last 90 days detailed scores
        - Green: Q1 (Goldilocks) & Q3 (Stagflation)
        - Blue: Q2 (Reflation) & Q4 (Deflation)
        
        **Strategy Performance:**
        - Backtests quadrant-based strategy
        - Long in Q1/Q3, flat in Q2/Q4
        - Compares vs buy & hold
        - Shows key performance metrics
        
        **Combined Dashboard:**
        - Overview of both analyses
        - Current position and performance
        """)

if __name__ == "__main__":
    main()

# ================================================================================================
# STRATEGY PERFORMANCE MODULE
# ================================================================================================

class StrategyPerformanceAnalysis:
    def __init__(self):
        self.strategy_name = "Quadrant Strategy"
        self.benchmark_name = "Buy & Hold"
    
    def calculate_strategy_performance(self, price_data: pd.DataFrame, daily_results: pd.DataFrame) -> Dict:
        """Calculate performance metrics for quadrant strategy vs buy & hold"""
        
        if price_data is None or 'BTC-USD' not in price_data.columns or daily_results is None:
            return None
        
        try:
            # Align price data with quadrant data
            btc_prices = price_data['BTC-USD'].copy()
            
            # Forward fill quadrant data to match price data length
            aligned_quadrants = pd.Series('Q2', index=btc_prices.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Forward fill quadrant assignments
            aligned_quadrants = aligned_quadrants.fillna(method='ffill')
            
            # CRITICAL FIX: Apply 1-day lag to avoid look-ahead bias
            # Shift quadrant signals forward by 1 day (lag the signal)
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            
            # Calculate daily returns
            btc_returns = btc_prices.pct_change().fillna(0)
            
            # Strategy: Long in Q1+Q3, Flat in Q2+Q4 (using lagged signals)
            strategy_positions = lagged_quadrants.isin(['Q1', 'Q3']).astype(int)
            strategy_returns = btc_returns * strategy_positions
            
            # Buy & Hold returns
            buyhold_returns = btc_returns
            
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
                'btc_prices': btc_prices,
                'signal_lag_applied': True  # Flag to indicate proper implementation
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
            name=self.strategy_name,
            line=dict(color='#00ff00', width=2)
        ))
        
        # Buy & Hold performance
        fig_performance.add_trace(go.Scatter(
            x=buyhold_metrics['cumulative_series'].index,
            y=buyhold_metrics['cumulative_series'].values * 100,
            mode='lines',
            name=self.benchmark_name,
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
        
        # Chart 2: Strategy Positions & BTC Price
        fig_positions = go.Figure()
        
        # BTC Price (secondary y-axis)
        fig_positions.add_trace(go.Scatter(
            x=performance_data['btc_prices'].index,
            y=performance_data['btc_prices'].values,
            mode='lines',
            name='BTC Price',
            line=dict(color='orange', width=1),
            yaxis='y2'
        ))
        
        # Strategy positions
        positions_for_plot = performance_data['strategy_positions'] * performance_data['btc_prices'].max() * 0.1
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
            title="Strategy Positions vs BTC Price",
            xaxis_title="Date",
            yaxis=dict(title="Position Signal", side="left"),
            yaxis2=dict(title="BTC Price (USD)", side="right", overlaying="y"),
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
        page_icon="📊", layout="wide", initial_sidebar_state="expanded")

    # Show dependency status
    if not YFINANCE_AVAILABLE:
        st.error("❌ **yfinance not available** - Quadrant analysis disabled")
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ **Plotly not available** - Using basic charts")

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
    st.sidebar.title("🎯 Dashboard Controls")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.selectbox("📊 Select Analysis", 
        ["Current Quadrant Analysis", "Strategy Performance", "Combined Dashboard"])

    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    lookback_days = st.sidebar.slider("Momentum Lookback (days)", 14, 50, 21)

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
            st.info("🟢 **Green**: Q1/Q3 quadrants AND above 50 EMA | 🔵 **Blue**: All other conditions | 🟠 **Orange dashed**: 50 EMA")
            
        else:
            # Fallback to basic streamlit chart (3 years)
            btc_with_ema = pd.DataFrame({
                'BTC Price': price_data['BTC-USD'],
                '50 EMA': price_data['BTC-USD'].rolling(window=50).mean()
            })
            st.line_chart(btc_with_ema)
            st.info("💡 Install plotly for color-coded quadrant chart with EMA filter")
        
        # Chart 2: Quadrant Scores (90 days only)
        st.subheader("Quadrant Scores - Last 90 Days")
        chart_data = last_90_days[['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score']]
        st.line_chart(chart_data)

    # Main content based on page selection
    if page == "Current Quadrant Analysis":
        st.markdown('<h1 class="main-header">📊 Current Quadrant Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("❌ Current Quadrant Analysis requires yfinance.")
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
                    <h3>🎯 Current Quadrant</h3>
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
                st.subheader("📈 Quadrant Distribution (90 days)")
                recent_quads = last_90_days['Primary_Quadrant'].value_counts()
                st.bar_chart(recent_quads)
                
                # Color legend
                st.markdown("""
                **Chart Color Legend:**
                - 🟢 **Green**: Q1 (Goldilocks) & Q3 (Stagflation)
                - 🔵 **Blue**: Q2 (Reflation) & Q4 (Deflation)
                """)
            
            # 90-day table
            st.subheader("📈 Last 90 Days Detailed View")
            
            display_df = last_90_days[['Primary_Quadrant', 'Primary_Score', 'Q1_Score', 
                                      'Q2_Score', 'Q3_Score', 'Q4_Score', 'Regime_Strength']].copy()
            display_df.index = display_df.index.strftime('%Y-%m-%d')
            display_df.columns = ['Quadrant', 'Score', 'Q1', 'Q2', 'Q3', 'Q4', 'Strength']
            
            st.dataframe(display_df.round(2), use_container_width=True, height=400)
            
        else:
            st.error("❌ Failed to load quadrant analysis data. Please check your internet connection.")

    elif page == "Strategy Performance":
        st.markdown('<h1 class="main-header">📈 Strategy Performance Analysis</h1>', unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("❌ Strategy Performance Analysis requires yfinance.")
            return
        
        # Load data
        with st.spinner("Loading strategy performance data..."):
            price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        
        if daily_results is not None and price_data is not None:
            # Calculate strategy performance
            strategy_analyzer = StrategyPerformanceAnalysis()
            performance_data = strategy_analyzer.calculate_strategy_performance(price_data, daily_results)
            
            if performance_data:
                strategy_metrics = performance_data['strategy_metrics']
                buyhold_metrics = performance_data['buyhold_metrics']
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Strategy Total Return", 
                             f"{strategy_metrics['total_return']:.1f}%",
                             delta=f"{strategy_metrics['total_return'] - buyhold_metrics['total_return']:.1f}% vs B&H")
                
                with col2:
                    st.metric("Sharpe Ratio", 
                             f"{strategy_metrics['sharpe_ratio']:.2f}",
                             delta=f"{strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", 
                             f"{strategy_metrics['max_drawdown']:.1f}%",
                             delta=f"{strategy_metrics['max_drawdown'] - buyhold_metrics['max_drawdown']:.1f}%")
                
                with col4:
                    st.metric("Time in Market", 
                             f"{performance_data['time_in_market']:.1f}%")
                
                # Performance charts
                strategy_analyzer.create_performance_charts(performance_data)
                
                # Detailed metrics comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Quadrant Strategy Metrics")
                    strategy_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                                  'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                        'Value': [f"{strategy_metrics['total_return']:.2f}",
                                 f"{strategy_metrics['annualized_return']:.2f}",
                                 f"{strategy_metrics['volatility']:.2f}",
                                 f"{strategy_metrics['sharpe_ratio']:.2f}",
                                 f"{strategy_metrics['max_drawdown']:.2f}",
                                 f"{strategy_metrics['win_rate']:.2f}"]
                    })
                    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("📊 Buy & Hold Metrics")
                    buyhold_df = pd.DataFrame({
                        'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                                  'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                        'Value': [f"{buyhold_metrics['total_return']:.2f}",
                                 f"{buyhold_metrics['annualized_return']:.2f}",
                                 f"{buyhold_metrics['volatility']:.2f}",
                                 f"{buyhold_metrics['sharpe_ratio']:.2f}",
                                 f"{buyhold_metrics['max_drawdown']:.2f}",
                                 f"{buyhold_metrics['win_rate']:.2f}"]
                    })
                    st.dataframe(buyhold_df, use_container_width=True, hide_index=True)
                
                # Strategy summary
                st.subheader("📋 Strategy Summary")
                
                outperformance = strategy_metrics['total_return'] - buyhold_metrics['total_return']
                risk_adjusted = strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']
                
                if outperformance > 0:
                    perf_color = "🟢"
                    perf_text = "outperformed"
                else:
                    perf_color = "🔴"
                    perf_text = "underperformed"
                
                st.markdown(f"""
                **Strategy Analysis:**
                - {perf_color} The Quadrant Strategy **{perf_text}** Buy & Hold by **{outperformance:+.1f}%**
                - **Risk-Adjusted Performance**: Sharpe ratio difference of **{risk_adjusted:+.2f}**
                - **Market Exposure**: Only **{performance_data['time_in_market']:.1f}%** of the time ({performance_data['long_days']} days long, {performance_data['flat_days']} days flat)
                - **Strategy Logic**: Long during Q1 (Goldilocks) and Q3 (Stagflation), flat during Q2 (Reflation) and Q4 (Deflation)
                """)
                
                # Technical implementation note
                st.info("""
                **📋 Strategy Implementation Notes:**
                
                **Signal Lag Applied**: This backtest uses **1-day lagged signals** to avoid look-ahead bias:
                - **Day T-1**: Calculate quadrant using momentum data through T-1
                - **Day T**: Apply that signal to Day T's trading (realistic implementation)
                - **No Cheating**: We never use "today's" price to make "today's" trading decision
                
                **Real Trading**: In practice, you'd calculate the quadrant at market close and apply the position the next day.
                """)
                
            else:
                st.error("❌ Failed to calculate strategy performance.")
        else:
            st.error("❌ Failed to load strategy performance data.")

    else:  # Combined Dashboard
        st.markdown('<h1 class="main-header">🚀 Combined Macro Flow Dashboard</h1>', unsafe_allow_html=True)
        
        # Load quadrant data
        if YFINANCE_AVAILABLE:
            with st.spinner("Loading quadrant analysis..."):
                price_data, daily_results, analyzer = load_quadrant_data(lookback_days)
        else:
            price_data, daily_results, analyzer = None, None, None
        
        if daily_results is not None:
            # Current status row - updated to use 90 days
            current_data = daily_results.tail(90).iloc[-1]
            current_quadrant = current_data['Primary_Quadrant']
            
            # Calculate quick strategy performance
            strategy_analyzer = StrategyPerformanceAnalysis()
            performance_data = strategy_analyzer.calculate_strategy_performance(price_data, daily_results)
            
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
                if performance_data:
                    strategy_return = performance_data['strategy_metrics']['total_return']
                    buyhold_return = performance_data['buyhold_metrics']['total_return']
                    outperformance = strategy_return - buyhold_return
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📈 Strategy Performance</h4>
                        <h3>{strategy_return:+.1f}%</h3>
                        <p>Outperformance: {outperformance:+.1f}%</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📈 Strategy Performance</h4>
                        <h3>Loading...</h3>
                        <p>Calculating returns</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col3:
                if performance_data:
                    time_in_market = performance_data['time_in_market']
                    position = "🟢 LONG" if current_quadrant in ['Q1', 'Q3'] else "🔵 FLAT"
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>🎯 Current Position</h4>
                        <h3>{position}</h3>
                        <p>Time in market: {time_in_market:.1f}%</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>🎯 Current Position</h4>
                        <h3>Loading...</h3>
                        <p>Analyzing position</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Main charts - updated to use 90 days
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if price_data is not None:
                    st.subheader("Bitcoin Price Trend - Last 90 Days")
                    st.line_chart(price_data['BTC-USD'].tail(90))
            
            with col2:
                if performance_data and PLOTLY_AVAILABLE:
                    st.subheader("Strategy vs Buy & Hold (Last 30 Days)")
                    last_30_strategy = performance_data['strategy_metrics']['cumulative_series'].tail(30)
                    last_30_buyhold = performance_data['buyhold_metrics']['cumulative_series'].tail(30)
                    
                    chart_data = pd.DataFrame({
                        'Strategy': (last_30_strategy.values - 1) * 100,
                        'Buy & Hold': (last_30_buyhold.values - 1) * 100
                    }, index=last_30_strategy.index)
                    
                    st.line_chart(chart_data)
                else:
                    st.info("Strategy performance chart loading...")
            
            # Quick stats - updated to use 90 days
            col1, col2 = st.columns(2)
            
            with col1:
                if daily_results is not None:
                    st.subheader("📈 Recent Quadrant Trend (Last 7 Days)")
                    recent_quads = daily_results.tail(7)['Primary_Quadrant'].value_counts()
                    st.bar_chart(recent_quads)
            
            with col2:
                st.subheader("🎯 Strategy Key Stats")
                if performance_data:
                    key_stats = pd.DataFrame({
                        'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
                        'Strategy': [f"{performance_data['strategy_metrics']['total_return']:.1f}%",
                                   f"{performance_data['strategy_metrics']['sharpe_ratio']:.2f}",
                                   f"{performance_data['strategy_metrics']['max_drawdown']:.1f}%",
                                   f"{performance_data['strategy_metrics']['win_rate']:.1f}%"],
                        'Buy & Hold': [f"{performance_data['buyhold_metrics']['total_return']:.1f}%",
                                     f"{performance_data['buyhold_metrics']['sharpe_ratio']:.2f}",
                                     f"{performance_data['buyhold_metrics']['max_drawdown']:.1f}%",
                                     f"{performance_data['buyhold_metrics']['win_rate']:.1f}%"]
                    })
                    st.dataframe(key_stats, use_container_width=True, hide_index=True)
                else:
                    st.info("Loading strategy statistics...")
            
        else:
            st.error("❌ Failed to load dashboard data. Please refresh the page.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Sources")
    st.sidebar.markdown("• **Quadrant Analysis**: Yahoo Finance")
    st.sidebar.markdown("• **Strategy Performance**: Calculated returns")
    st.sidebar.markdown("• **Refresh Rate**: 5 minutes")
    
    # Status indicators
    st.sidebar.markdown("### 🔧 System Status")
    if YFINANCE_AVAILABLE:
        st.sidebar.markdown("• ✅ Yahoo Finance: Ready")
    else:
        st.sidebar.markdown("• ❌ Yahoo Finance: Missing")
    
    if PLOTLY_AVAILABLE:
        st.sidebar.markdown("• ✅ Plotly Charts: Ready")
    else:
        st.sidebar.markdown("• ⚠️ Plotly Charts: Basic mode")
    
    # Instructions
    with st.sidebar.expander("📖 How to Use"):
        st.markdown("""
        **Quadrant Analysis:**
        - Shows current market regime (Growth/Inflation)
        - BTC chart colored by quadrant periods
        - Last 90 days detailed scores
        - Green: Q1 (Goldilocks) & Q3 (Stagflation)
        - Blue: Q2 (Reflation) & Q4 (Deflation)
        
        **Strategy Performance:**
        - Backtests quadrant-based strategy
        - Long in Q1/Q3, flat in Q2/Q4
        - Compares vs buy & hold
        - Shows key performance metrics
        
        **Combined Dashboard:**
        - Overview of both analyses
        - Current position and performance
        """)

if __name__ == "__main__":
    main()
