#!/usr/bin/env python3
"""
Quadrant Analysis Module for Macro Dashboard
Handles market regime classification and quadrant scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import streamlit as st

from config import (
    AssetClassification, CORE_ASSETS, Q1_ASSETS, Q2_ASSETS, 
    Q3_ASSETS, Q4_ASSETS, QUADRANT_DESCRIPTIONS, DEFAULT_LOOKBACK_DAYS
)
from data_fetcher import DataFetcher

class CurrentQuadrantAnalysis:
    """Handles quadrant analysis and market regime classification"""
    
    def __init__(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS):
        self.lookback_days = lookback_days
        self.asset_classifications = self._initialize_asset_classifications()
        self.data_fetcher = DataFetcher()
    
    def _initialize_asset_classifications(self) -> Dict[str, AssetClassification]:
        """Initialize asset classifications by quadrant"""
        classifications = {}
        
        # Add all assets from different quadrants
        for symbol, name, quad in Q1_ASSETS + Q2_ASSETS + Q3_ASSETS + Q4_ASSETS:
            classifications[symbol] = AssetClassification(symbol, name, quad)
        
        return classifications
    
    def fetch_recent_data(self, days_back: int = 200) -> pd.DataFrame:
        """Fetch recent data for all core assets"""
        symbols = list(CORE_ASSETS.keys())
        return self.data_fetcher.fetch_recent_data(symbols, days_back)
    
    def calculate_daily_momentum(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily momentum for all assets"""
        return self.data_fetcher.calculate_momentum(price_data, self.lookback_days)
    
    def calculate_daily_quadrant_scores(self, momentum_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily quadrant scores based on asset momentum"""
        quadrant_scores = pd.DataFrame(index=momentum_data.index)
        
        # Initialize score and count columns for each quadrant
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            quadrant_scores[f'{quad}_Score'] = 0.0
            quadrant_scores[f'{quad}_Count'] = 0
        
        # Calculate daily scores for each quadrant
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
            
            # Store daily scores and counts
            for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
                quadrant_scores.loc[date, f'{quad}_Score'] = daily_scores[quad]
                quadrant_scores.loc[date, f'{quad}_Count'] = daily_counts[quad]
        
        return quadrant_scores
    
    def determine_daily_quadrant(self, quadrant_scores: pd.DataFrame) -> pd.DataFrame:
        """Determine the primary quadrant for each day"""
        results = pd.DataFrame(index=quadrant_scores.index)
        
        # Calculate normalized scores for each quadrant
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            score_col = f'{quad}_Score'
            count_col = f'{quad}_Count'
            results[f'{quad}_Normalized'] = np.where(
                quadrant_scores[count_col] > 0,
                quadrant_scores[score_col] / quadrant_scores[count_col], 
                0
            )
        
        # Determine primary quadrant (highest normalized score)
        quad_cols = ['Q1_Normalized', 'Q2_Normalized', 'Q3_Normalized', 'Q4_Normalized']
        results['Primary_Quadrant'] = results[quad_cols].idxmax(axis=1).str.replace('_Normalized', '')
        results['Primary_Score'] = results[quad_cols].max(axis=1)
        
        # Calculate secondary score (second highest)
        results['Secondary_Score'] = results[quad_cols].apply(
            lambda row: row.nlargest(2).iloc[1] if len(row.nlargest(2)) > 1 else 0, 
            axis=1
        )
        
        # Calculate confidence (ratio of primary to secondary score)
        results['Confidence'] = np.where(
            results['Secondary_Score'] > 0,
            results['Primary_Score'] / results['Secondary_Score'], 
            float('inf')
        )
        
        # Classify regime strength based on confidence
        results['Regime_Strength'] = pd.cut(
            results['Confidence'],
            bins=[0, 1.2, 1.8, float('inf')],
            labels=['Weak', 'Medium', 'Strong']
        )
        
        # Store original scores for reference
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            results[f'{quad}_Score'] = results[f'{quad}_Normalized']
        
        return results
    
    def analyze_current_quadrant_and_30_days(self) -> pd.DataFrame:
        """Analyze current quadrant and provide 30-day view"""
        try:
            # Fetch recent data
            price_data = self.fetch_recent_data()
            if price_data.empty:
                st.error("Failed to fetch price data")
                return pd.DataFrame()
            
            # Calculate momentum
            momentum_data = self.calculate_daily_momentum(price_data)
            if momentum_data.empty:
                st.error("Failed to calculate momentum data")
                return pd.DataFrame()
            
            # Calculate quadrant scores
            quadrant_scores = self.calculate_daily_quadrant_scores(momentum_data)
            if quadrant_scores.empty:
                st.error("Failed to calculate quadrant scores")
                return pd.DataFrame()
            
            # Determine daily quadrants
            results = self.determine_daily_quadrant(quadrant_scores)
            
            # Clean and validate results
            results = self.data_fetcher.clean_and_validate_data(results)
            
            return results
            
        except Exception as e:
            st.error(f"Error in quadrant analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_quadrant_summary(self, results: pd.DataFrame) -> Dict:
        """Get summary of quadrant analysis results"""
        if results.empty:
            return {}
        
        latest = results.iloc[-1]
        
        summary = {
            'current_quadrant': latest.get('Primary_Quadrant', 'Unknown'),
            'current_score': latest.get('Primary_Score', 0),
            'confidence': latest.get('Confidence', 0),
            'strength': latest.get('Regime_Strength', 'Unknown'),
            'date': latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name),
            'total_days': len(results)
        }
        
        return summary
    
    def get_quadrant_descriptions(self) -> Dict[str, str]:
        """Get quadrant descriptions"""
        return QUADRANT_DESCRIPTIONS.copy()
