#!/usr/bin/env python3
"""
Main Dashboard Module for Macro Dashboard
Integrates all modules and provides the Streamlit interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Handle imports with proper error checking
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    PLOTLY_AVAILABLE = False

# Import our modules
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, REFRESH_RATE,
    QUADRANT_DESCRIPTIONS, DEFAULT_TOP_N_TOKENS
)
from quadrant_analysis import CurrentQuadrantAnalysis
from axe_list_generator import AxeListGenerator
from strategy_performance import StrategyPerformanceAnalysis
from data_fetcher import DataFetcher

warnings.filterwarnings('ignore')

class MacroDashboard:
    """Main dashboard class that integrates all modules"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        
        # Initialize modules
        self.quadrant_analyzer = CurrentQuadrantAnalysis()
        self.axe_generator = AxeListGenerator()
        self.performance_analyzer = StrategyPerformanceAnalysis()
        self.data_fetcher = DataFetcher()
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout=LAYOUT,
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #00ff88;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #333;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'axe_data' not in st.session_state:
            st.session_state['axe_data'] = None
        if 'baseline_asset' not in st.session_state:
            st.session_state['baseline_asset'] = None
        if 'last_refresh' not in st.session_state:
            st.session_state['last_refresh'] = datetime.now()
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">📊 Crypto Macro Flow Dashboard</h1>', unsafe_allow_html=True)
        
        # Auto-refresh info
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(f"🔄 Auto-refreshes every {REFRESH_RATE//60} minutes | Last update: {st.session_state['last_refresh'].strftime('%H:%M:%S')}")
    
    def render_sidebar(self):
        """Render the sidebar with controls and info"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Analysis parameters
        st.sidebar.subheader("Analysis Settings")
        lookback_days = st.sidebar.slider("Lookback Days", 7, 100, 21, help="Days for momentum calculation")
        top_n_tokens = st.sidebar.slider("Top N Tokens", 5, 20, DEFAULT_TOP_N_TOKENS, help="Number of top tokens to display")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            st.session_state['last_refresh'] = datetime.now()
            st.experimental_rerun()
        
        st.sidebar.markdown("---")
        
        # Data sources info
        st.sidebar.markdown("### 📊 Data Sources")
        st.sidebar.markdown("• **Quadrant Analysis**: Yahoo Finance")
        st.sidebar.markdown("• **Axe List**: CoinGecko + Binance")
        st.sidebar.markdown("• **Strategy Performance**: Calculated returns")
        
        # System status
        st.sidebar.markdown("### 🔧 System Status")
        if PLOTLY_AVAILABLE:
            st.sidebar.markdown("• Plotly Charts: ✅ Ready")
        else:
            st.sidebar.markdown("• Plotly Charts: ❌ Missing")
        
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
            
            **Axe List:**
            - Top performing crypto tokens
            - Ranked by momentum score
            - 7-day and 30-day performance
            """)
    
    def render_quadrant_analysis(self):
        """Render the quadrant analysis section"""
        st.header("🎯 Market Quadrant Analysis")
        
        # Run quadrant analysis
        with st.spinner("Analyzing market quadrants..."):
            results = self.quadrant_analyzer.analyze_current_quadrant_and_30_days()
        
        if results is None or results.empty:
            st.error("Failed to load quadrant analysis data")
            return
        
        # Get current quadrant info
        summary = self.quadrant_analyzer.get_quadrant_summary(results)
        current_quadrant = summary.get('current_quadrant', 'Unknown')
        current_score = summary.get('current_score', 0)
        confidence = summary.get('confidence', 0)
        strength = summary.get('strength', 'Unknown')
        
        # Display current quadrant
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Quadrant", current_quadrant)
        with col2:
            st.metric("Score", f"{current_score:.2f}")
        with col3:
            st.metric("Confidence", f"{confidence:.2f}" if confidence != float('inf') else "Very High")
        with col4:
            st.metric("Strength", strength)
        
        # Quadrant description
        quadrant_desc = QUADRANT_DESCRIPTIONS.get(current_quadrant, "Unknown")
        st.info(f"**{current_quadrant}**: {quadrant_desc}")
        
        # Recent quadrant history
        st.subheader("📈 Recent Quadrant History")
        recent_results = results.tail(90)  # Last 90 days
        
        if not recent_results.empty:
            # Create quadrant timeline chart
            fig = go.Figure()
            
            # Color coding for quadrants
            colors = {'Q1': '#00ff88', 'Q3': '#00ff88', 'Q2': '#0088ff', 'Q4': '#0088ff'}
            
            for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = recent_results['Primary_Quadrant'] == quadrant
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=recent_results[mask].index,
                        y=recent_results[mask][f'{quadrant}_Score'],
                        mode='markers',
                        name=quadrant,
                        marker=dict(color=colors[quadrant], size=8),
                        hovertemplate=f'{quadrant}<br>Score: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title="Quadrant Scores (Last 90 Days)",
                xaxis_title="Date",
                yaxis_title="Score",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores table
        st.subheader("📊 Detailed Scores")
        score_cols = ['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score', 'Primary_Quadrant', 'Primary_Score', 'Confidence', 'Regime_Strength']
        display_cols = [col for col in score_cols if col in recent_results.columns]
        
        if display_cols:
            st.dataframe(
                recent_results[display_cols].tail(30),
                use_container_width=True,
                height=300
            )
    
    def render_strategy_performance(self):
        """Render the strategy performance section"""
        st.header("📊 Strategy Performance Analysis")
        
        # Get quadrant results for performance calculation
        results = self.quadrant_analyzer.analyze_current_quadrant_and_30_days()
        if results is None or results.empty:
            st.error("Quadrant analysis required for strategy performance")
            return
        
        # Get price data for BTC and ETH
        price_data = self.data_fetcher.fetch_recent_data(['BTC-USD', 'ETH-USD'])
        if price_data.empty:
            st.error("Failed to fetch price data for performance analysis")
            return
        
        # Calculate portfolio performance
        with st.spinner("Calculating strategy performance..."):
            performance_data = self.performance_analyzer.calculate_strategy_performance(
                price_data, results, is_portfolio=True
            )
        
        if not performance_data:
            st.error("Failed to calculate strategy performance")
            return
        
        # Display performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        strategy_metrics = performance_data['strategy_metrics']
        buyhold_metrics = performance_data['buyhold_metrics']
        
        with col1:
            st.metric("Strategy Return", f"{strategy_metrics.get('total_return', 0):.1f}%")
        with col2:
            st.metric("Buy & Hold Return", f"{buyhold_metrics.get('total_return', 0):.1f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{strategy_metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{strategy_metrics.get('max_drawdown', 0):.1f}%")
        
        # Performance comparison chart
        st.subheader("📈 Performance Comparison")
        chart_data = self.performance_analyzer.generate_performance_chart_data(performance_data)
        
        if chart_data is not None:
            fig = go.Figure()
            
            for col in chart_data.columns:
                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Strategy vs Buy & Hold Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (Base 100)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics table
        st.subheader("📋 Performance Metrics")
        if strategy_metrics and buyhold_metrics:
            metrics_df = pd.DataFrame({
                'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Volatility'],
                'Strategy': [
                    f"{strategy_metrics.get('total_return', 0):.1f}%",
                    f"{strategy_metrics.get('sharpe_ratio', 0):.2f}",
                    f"{strategy_metrics.get('max_drawdown', 0):.1f}%",
                    f"{strategy_metrics.get('win_rate', 0):.1f}%",
                    f"{strategy_metrics.get('volatility', 0):.1f}%"
                ],
                'Buy & Hold': [
                    f"{buyhold_metrics.get('total_return', 0):.1f}%",
                    f"{buyhold_metrics.get('sharpe_ratio', 0):.2f}",
                    f"{buyhold_metrics.get('max_drawdown', 0):.1f}%",
                    f"{buyhold_metrics.get('win_rate', 0):.1f}%",
                    f"{buyhold_metrics.get('volatility', 0):.1f}%"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    def render_axe_list(self):
        """Render the axe list section"""
        st.header("🚀 Top Performing Tokens (Axe List)")
        
        # Display existing axe list if available
        if st.session_state['axe_data'] is not None:
            axe_data = st.session_state['axe_data']
            
            st.success(f"✅ Axe list loaded! Found {len(axe_data)} tokens")
            
            # Display top tokens
            st.subheader("🏆 Top Performers")
            display_cols = ['symbol', 'name', 'price', 'day_return', 'week_return', 'month_return', 'momentum_score']
            available_cols = [col for col in display_cols if col in axe_data.columns]
            
            if available_cols:
                st.dataframe(
                    axe_data[available_cols].head(8),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Performance chart
            if len(axe_data) >= 8:
                st.subheader("📊 Performance Chart")
                top_8 = axe_data.head(8)
                
                chart_data = pd.DataFrame({
                    'Token': top_8['symbol'],
                    '7d Return': top_8['week_return'],
                    '30d Return': top_8['month_return']
                }).set_index('Token')
                
                st.bar_chart(chart_data)
            
            # Show baseline asset
            if 'baseline_asset' in st.session_state:
                st.info(f"🎯 Baseline: {st.session_state['baseline_asset']}")
            else:
                st.info("🎯 Baseline: BTC/ETH (auto-detected)")
        
        # Generate/Refresh button
        if st.button("🚀 Generate/Refresh Axe List", type="primary"):
            with st.spinner("Generating axe list..."):
                try:
                    axe_data = self.axe_generator.run_analysis(DEFAULT_TOP_N_TOKENS)
                    
                    if axe_data is not None and not axe_data.empty:
                        st.session_state['axe_data'] = axe_data
                        if hasattr(self.axe_generator, 'last_baseline'):
                            st.session_state['baseline_asset'] = self.axe_generator.last_baseline
                        st.success(f"✅ Axe list updated! Found {len(axe_data)} tokens")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Failed to generate axe list")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    def run(self):
        """Main method to run the dashboard"""
        try:
            # Render components
            self.render_header()
            self.render_sidebar()
            
            # Main content
            self.render_quadrant_analysis()
            st.markdown("---")
            self.render_strategy_performance()
            st.markdown("---")
            self.render_axe_list()
            
            # Footer
            st.markdown("---")
            st.markdown("*Dashboard auto-refreshes every 5 minutes*")
            
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            st.exception(e)

def main():
    """Main entry point"""
    try:
        dashboard = MacroDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
