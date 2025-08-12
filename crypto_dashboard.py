#!/usr/bin/env python3
"""
Crypto Macro Flow Dashboard - Minimal Demo Version
Works with just streamlit, pandas, numpy - no external APIs needed
Perfect for testing the interface before installing other dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Crypto Macro Flow Dashboard - Demo",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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

def generate_demo_quadrant_data():
    """Generate demo quadrant data"""
    dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='D')
    
    # Generate mock BTC price data
    np.random.seed(42)
    price_base = 40000
    price_data = []
    current_price = price_base
    
    for i in range(len(dates)):
        # Add some realistic price movement
        daily_change = np.random.normal(0.001, 0.03)  # 0.1% mean, 3% std
        current_price *= (1 + daily_change)
        price_data.append(current_price)
    
    btc_df = pd.DataFrame({'BTC-USD': price_data}, index=dates)
    
    # Generate mock quadrant scores
    quadrant_data = []
    for i, date in enumerate(dates):
        # Simulate changing market regimes
        cycle_pos = (i / len(dates)) * 4  # 4 major cycles over time period
        
        if cycle_pos < 1:  # Q1 dominant period
            q1, q2, q3, q4 = 15 + np.random.normal(0, 5), -5 + np.random.normal(0, 3), -10 + np.random.normal(0, 3), -8 + np.random.normal(0, 3)
            primary_quad = 'Q1'
        elif cycle_pos < 2:  # Q2 dominant period
            q1, q2, q3, q4 = 5 + np.random.normal(0, 3), 18 + np.random.normal(0, 5), -5 + np.random.normal(0, 3), -12 + np.random.normal(0, 3)
            primary_quad = 'Q2'
        elif cycle_pos < 3:  # Q3 dominant period
            q1, q2, q3, q4 = -8 + np.random.normal(0, 3), -2 + np.random.normal(0, 3), 20 + np.random.normal(0, 5), -5 + np.random.normal(0, 3)
            primary_quad = 'Q3'
        else:  # Q4 dominant period
            q1, q2, q3, q4 = -10 + np.random.normal(0, 3), -8 + np.random.normal(0, 3), -5 + np.random.normal(0, 3), 16 + np.random.normal(0, 5)
            primary_quad = 'Q4'
        
        # Add some noise and regime uncertainty
        if np.random.random() < 0.15:  # 15% chance of regime transition
            scores = [q1, q2, q3, q4]
            max_idx = np.argmax(scores)
            primary_quad = ['Q1', 'Q2', 'Q3', 'Q4'][max_idx]
        
        confidence = np.random.uniform(1.2, 3.5)
        strength = 'Strong' if confidence > 2.5 else 'Medium' if confidence > 1.8 else 'Weak'
        
        quadrant_data.append({
            'Q1_Score': q1, 'Q2_Score': q2, 'Q3_Score': q3, 'Q4_Score': q4,
            'Primary_Quadrant': primary_quad,
            'Primary_Score': max(q1, q2, q3, q4),
            'Confidence': confidence,
            'Regime_Strength': strength
        })
    
    quad_df = pd.DataFrame(quadrant_data, index=dates)
    
    return btc_df, quad_df

def generate_demo_axe_data():
    """Generate demo axe list data"""
    tokens = [
        ('Ethereum', 'ETH', 380000000000, 2),
        ('Solana', 'SOL', 95000000000, 5),
        ('Cardano', 'ADA', 18000000000, 8),
        ('Polygon', 'MATIC', 9000000000, 13),
        ('Chainlink', 'LINK', 15000000000, 15),
        ('Avalanche', 'AVAX', 12000000000, 18),
        ('Polkadot', 'DOT', 8000000000, 20),
        ('Uniswap', 'UNI', 7000000000, 22),
        ('Litecoin', 'LTC', 6500000000, 25),
        ('Near Protocol', 'NEAR', 4500000000, 28)
    ]
    
    demo_data = []
    for i, (name, symbol, market_cap, rank) in enumerate(tokens):
        # Generate realistic performance metrics
        np.random.seed(42 + i)
        
        ratio_vs_ma = np.random.normal(8, 15)  # Some positive, some negative
        week_return = np.random.normal(3, 12)
        month_return = np.random.normal(15, 25)
        above_ma50 = ratio_vs_ma > 0
        above_ma20 = np.random.choice([True, False], p=[0.7, 0.3])
        token_outperforming = ratio_vs_ma > 5
        
        demo_data.append({
            'name': name,
            'symbol': symbol,
            'ratio_vs_ma': ratio_vs_ma,
            'market_cap': market_cap,
            'market_cap_rank': rank,
            'week_return': week_return,
            'month_return': month_return,
            'above_ma50': above_ma50,
            'above_ma20': above_ma20,
            'token_outperforming': token_outperforming
        })
    
    df = pd.DataFrame(demo_data)
    df = df.sort_values('ratio_vs_ma', ascending=False)
    return df

def main():
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.sidebar.title("🎯 Dashboard Controls")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.warning("📋 **DEMO MODE**\nUsing simulated data for interface testing")

    # Navigation
    page = st.sidebar.selectbox("📊 Select Analysis", 
        ["Current Quadrant Analysis", "Axe List Generator", "Combined Dashboard"])

    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    lookback_days = st.sidebar.slider("Momentum Lookback (days)", 14, 50, 21)
    top_n_tokens = st.sidebar.slider("Top N Tokens for Axe List", 5, 20, 10)

    # Generate demo data
    btc_data, quad_data = generate_demo_quadrant_data()
    axe_data = generate_demo_axe_data()
    
    quadrant_descriptions = {
        'Q1': 'Growth ↑, Inflation ↓ (Goldilocks)',
        'Q2': 'Growth ↑, Inflation ↑ (Reflation)', 
        'Q3': 'Growth ↓, Inflation ↑ (Stagflation)',
        'Q4': 'Growth ↓, Inflation ↓ (Deflation)'
    }

    if page == "Current Quadrant Analysis":
        st.markdown('<h1 class="main-header">📊 Current Quadrant Analysis</h1>', unsafe_allow_html=True)
        
        # Current quadrant info
        last_30_days = quad_data.tail(30)
        current_data = last_30_days.iloc[-1]
        current_quadrant = current_data['Primary_Quadrant']
        current_score = current_data['Primary_Score']
        
        # Display current quadrant
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="quadrant-card">
                <h3>🎯 Current Quadrant</h3>
                <h1>{current_quadrant}</h1>
                <p>{quadrant_descriptions[current_quadrant]}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            prev_score = last_30_days['Primary_Score'].iloc[-2]
            st.metric("Primary Score", f"{current_score:.2f}", delta=f"{current_score - prev_score:.2f}")
        
        with col3:
            confidence_val = current_data['Confidence']
            st.metric("Confidence", f"{confidence_val:.2f}")
        
        with col4:
            st.metric("Regime Strength", current_data['Regime_Strength'])
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Bitcoin Price (Demo Data)")
            st.line_chart(btc_data['BTC-USD'])
            
            # Show quadrant periods
            recent_quads = last_30_days['Primary_Quadrant'].value_counts()
            st.subheader("Recent Quadrant Distribution (Last 30 Days)")
            st.bar_chart(recent_quads)
        
        with col2:
            st.subheader("Quadrant Scores - Last 30 Days")
            chart_data = last_30_days[['Q1_Score', 'Q2_Score', 'Q3_Score', 'Q4_Score']]
            st.line_chart(chart_data)
        
        # 30-day table
        st.subheader("📈 Last 30 Days Detailed View")
        
        display_df = last_30_days[['Primary_Quadrant', 'Primary_Score', 'Q1_Score', 
                                  'Q2_Score', 'Q3_Score', 'Q4_Score', 'Regime_Strength']].copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d')
        display_df.columns = ['Quadrant', 'Score', 'Q1', 'Q2', 'Q3', 'Q4', 'Strength']
        
        st.dataframe(display_df.round(2), use_container_width=True)

    elif page == "Axe List Generator":
        st.markdown('<h1 class="main-header">🎯 Axe List Generator</h1>', unsafe_allow_html=True)
        
        st.info("🎯 **Demo Baseline**: BTC (simulated analysis)")
        
        # Top performers metrics
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
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10: Ratio vs 50-day MA")
            chart_data = axe_data.head(10).set_index('symbol')['ratio_vs_ma']
            st.bar_chart(chart_data)
        
        with col2:
            st.subheader("Week vs Month Returns")
            chart_data = axe_data[['week_return', 'month_return']].head(10)
            # Create a simple correlation view
            st.scatter_chart(chart_data)
        
        # Detailed table
        st.subheader("🏆 Top Performers Detailed View")
        
        display_cols = ['name', 'symbol', 'ratio_vs_ma', 'market_cap_rank', 
                       'week_return', 'month_return', 'above_ma50', 'above_ma20']
        display_df = axe_data[display_cols].copy()
        display_df.columns = ['Name', 'Symbol', 'Ratio vs MA (%)', 'MCap Rank', 
                             'Week Return (%)', 'Month Return (%)', 'Above 50MA', 'Above 20MA']
        
        st.dataframe(display_df.round(2), use_container_width=True)

    else:  # Combined Dashboard
        st.markdown('<h1 class="main-header">🚀 Combined Macro Flow Dashboard</h1>', unsafe_allow_html=True)
        
        # Current status row
        current_data = quad_data.tail(30).iloc[-1]
        current_quadrant = current_data['Primary_Quadrant']
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f'''
            <div class="quadrant-card">
                <h4>Current Regime</h4>
                <h2>{current_quadrant}</h2>
                <p>{quadrant_descriptions[current_quadrant]}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            top_token = axe_data.iloc[0]
            st.markdown(f'''
            <div class="metric-card">
                <h4>🏆 Top Axe</h4>
                <h3>{top_token['name']}</h3>
                <p>Ratio vs MA: {top_token['ratio_vs_ma']:+.1f}%</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            outperforming = axe_data['token_outperforming'].sum()
            st.markdown(f'''
            <div class="metric-card">
                <h4>📊 Market Strength</h4>
                <h3>{outperforming}/{len(axe_data)}</h3>
                <p>Tokens outperforming baseline</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Main charts
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Bitcoin Price Trend")
            st.line_chart(btc_data['BTC-USD'].tail(365))  # Last year
        
        with col2:
            st.subheader("Top 8 Tokens: Ratio vs MA")
            chart_data = axe_data.head(8).set_index('symbol')['ratio_vs_ma']
            st.bar_chart(chart_data)
        
        # Quick stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Recent Quadrant Trend")
            recent_quads = quad_data.tail(7)['Primary_Quadrant'].value_counts()
            st.bar_chart(recent_quads)
        
        with col2:
            st.subheader("🎯 Top 5 Axe List")
            top_5 = axe_data.head(5)[['name', 'ratio_vs_ma', 'month_return']]
            top_5.columns = ['Token', 'Ratio vs MA (%)', 'Month Return (%)']
            st.dataframe(top_5.round(1), use_container_width=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Demo Mode Info")
    st.sidebar.markdown("• **Data**: Simulated/Demo")
    st.sidebar.markdown("• **Purpose**: Interface Testing")
    st.sidebar.markdown("• **Next Step**: Install dependencies for real data")
    
    st.sidebar.markdown("### 🚀 Install for Real Data")
    st.sidebar.code("pip install yfinance requests plotly")
    
    # Instructions
    with st.sidebar.expander("📖 About This Demo"):
        st.markdown("""
        **Demo Features:**
        - Simulated quadrant analysis data
        - Mock crypto token performance
        - Full interface functionality
        - No external dependencies needed
        
        **Real Version Adds:**
        - Live Yahoo Finance data
        - CoinGecko + Binance APIs
        - Interactive Plotly charts
        - Real-time analysis
        """)

if __name__ == "__main__":
    main()
