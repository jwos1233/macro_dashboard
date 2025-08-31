#!/usr/bin/env python3
"""
Strategy Performance Module for Macro Dashboard
Handles backtesting and performance analysis of trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import streamlit as st

from config import (
    STRATEGY_NAME, BENCHMARK_NAME, RISK_FREE_RATE, EMA_PERIOD
)

class StrategyPerformanceAnalysis:
    """Analyzes performance of trading strategies vs benchmarks"""
    
    def __init__(self):
        self.strategy_name = STRATEGY_NAME
        self.benchmark_name = BENCHMARK_NAME
    
    def calculate_strategy_performance(
        self, 
        price_data: pd.DataFrame, 
        daily_results: pd.DataFrame, 
        crypto_symbol: str = 'BTC-USD', 
        is_portfolio: bool = False
    ) -> Optional[Dict]:
        """
        Calculate performance metrics for quadrant strategy vs buy & hold
        
        Args:
            price_data: DataFrame with price data
            daily_results: DataFrame with quadrant analysis results
            crypto_symbol: Symbol for single crypto analysis
            is_portfolio: Whether to analyze portfolio or single asset
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Validate inputs
            if is_portfolio:
                if 'BTC-USD' not in price_data.columns or 'ETH-USD' not in price_data.columns:
                    st.error("Portfolio analysis requires both BTC and ETH data")
                    return None
            else:
                if price_data is None or crypto_symbol not in price_data.columns or daily_results is None:
                    st.error("Single crypto analysis requires valid price data and quadrant results")
                    return None
            
            # Prepare price data
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
            crypto_50ema = crypto_prices.rolling(window=EMA_PERIOD).mean()
            
            # Align quadrant data with price data
            aligned_quadrants = pd.Series('Q2', index=crypto_prices.index)
            for date in daily_results.index:
                if date in aligned_quadrants.index:
                    aligned_quadrants[date] = daily_results.loc[date, 'Primary_Quadrant']
            
            # Forward fill quadrant assignments
            aligned_quadrants = aligned_quadrants.fillna(method='ffill')
            
            # Apply 1-day lag to avoid look-ahead bias
            lagged_quadrants = aligned_quadrants.shift(1).fillna('Q2')
            lagged_50ema = crypto_50ema.shift(1).fillna(crypto_prices.iloc[0])
            
            # Calculate daily returns
            crypto_returns = crypto_prices.pct_change().fillna(0)
            
            # Strategy: Long in Q1+Q3 AND above 50 EMA, Flat otherwise
            favorable_quadrant = lagged_quadrants.isin(['Q1', 'Q3'])
            above_ema = crypto_prices > lagged_50ema
            strategy_positions = (favorable_quadrant & above_ema).astype(int)
            strategy_returns = crypto_returns * strategy_positions
            
            # Buy & Hold returns
            buyhold_returns = crypto_returns
            
            # Calculate cumulative performance
            strategy_cumulative = (1 + strategy_returns).cumprod()
            buyhold_cumulative = (1 + buyhold_returns).cumprod()
            
            # Calculate performance metrics
            strategy_metrics = self._calculate_metrics(strategy_returns, self.strategy_name, strategy_positions)
            buyhold_metrics = self._calculate_metrics(buyhold_returns, self.benchmark_name)
            
            # Prepare results
            results = {
                'strategy_metrics': strategy_metrics,
                'buyhold_metrics': buyhold_metrics,
                'strategy_cumulative': strategy_cumulative,
                'buyhold_cumulative': buyhold_cumulative,
                'strategy_positions': strategy_positions,
                'quadrants': aligned_quadrants,
                'crypto_name': crypto_name,
                'analysis_period': {
                    'start': crypto_prices.index[0].strftime('%Y-%m-%d'),
                    'end': crypto_prices.index[-1].strftime('%Y-%m-%d'),
                    'days': len(crypto_prices)
                }
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error calculating strategy performance: {e}")
            return None
    
    def _calculate_metrics(self, returns_series: pd.Series, name: str, positions: Optional[pd.Series] = None) -> Dict:
        """Calculate performance metrics for a returns series"""
        if len(returns_series) == 0 or returns_series.std() == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns_series).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
        
        # Calculate drawdowns
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win rate calculation
        if name == self.strategy_name and positions is not None:
            # Only count days when strategy was actually long
            active_returns = returns_series[positions == 1]
            positive_days = (active_returns > 0).sum()
            total_days = len(active_returns)
        else:
            positive_days = (returns_series > 0).sum()
            total_days = len(returns_series[returns_series != 0])
        
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        
        # Additional metrics
        avg_return = returns_series.mean() * 252
        downside_deviation = returns_series[returns_series < 0].std() * np.sqrt(252)
        sortino_ratio = (annualized_return - RISK_FREE_RATE) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate,
            'avg_return': avg_return * 100,
            'sortino_ratio': sortino_ratio,
            'total_days': total_days,
            'positive_days': positive_days
        }
    
    def get_performance_summary(self, results: Dict) -> Dict:
        """Get summary of performance analysis"""
        if not results:
            return {}
        
        summary = {
            'strategy_name': self.strategy_name,
            'benchmark_name': self.benchmark_name,
            'analysis_period': results.get('analysis_period', {}),
            'crypto_name': results.get('crypto_name', 'Unknown'),
            'strategy_metrics': results.get('strategy_metrics', {}),
            'benchmark_metrics': results.get('buyhold_metrics', {}),
            'outperformance': 0
        }
        
        # Calculate outperformance
        strategy_return = summary['strategy_metrics'].get('total_return', 0)
        benchmark_return = summary['benchmark_metrics'].get('total_return', 0)
        summary['outperformance'] = strategy_return - benchmark_return
        
        return summary
    
    def generate_performance_chart_data(self, results: Dict) -> Optional[pd.DataFrame]:
        """Generate data for performance comparison charts"""
        if not results:
            return None
        
        try:
            strategy_cumulative = results.get('strategy_cumulative')
            buyhold_cumulative = results.get('buyhold_cumulative')
            
            if strategy_cumulative is None or buyhold_cumulative is None:
                return None
            
            # Create comparison DataFrame
            chart_data = pd.DataFrame({
                'Strategy': strategy_cumulative,
                'Buy & Hold': buyhold_cumulative
            })
            
            return chart_data
            
        except Exception as e:
            st.error(f"Error generating chart data: {e}")
            return None
