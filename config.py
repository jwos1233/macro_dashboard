#!/usr/bin/env python3
"""
Configuration file for Macro Dashboard
Centralizes all constants, settings, and asset classifications
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ================================================================================================
# ASSET CLASSIFICATIONS
# ================================================================================================

@dataclass
class AssetClassification:
    symbol: str
    name: str
    primary_quadrant: str
    secondary_quadrant: str = None
    asset_type: str = ""
    weight: float = 1.0

# Core assets for quadrant analysis
CORE_ASSETS = {
    'QQQ': 'NASDAQ 100 (Growth)', 
    'VUG': 'Vanguard Growth ETF',
    'IWM': 'Russell 2000 (Small Caps)', 
    'BTC-USD': 'Bitcoin (BTC)',
    'ETH-USD': 'Ethereum (ETH)',
    'XLE': 'Energy Sector ETF', 
    'DBC': 'Broad Commodities ETF',
    'GLD': 'Gold ETF', 
    'LIT': 'Lithium & Battery Tech ETF',
    'TLT': '20+ Year Treasury Bonds', 
    'XLU': 'Utilities Sector ETF',
    'VIXY': 'Short-Term VIX Futures ETF',
}

# Asset classifications by quadrant
Q1_ASSETS = [
    ('QQQ', 'NASDAQ 100 (Growth)', 'Q1'), 
    ('VUG', 'Vanguard Growth ETF', 'Q1'),
    ('IWM', 'Russell 2000 (Small Caps)', 'Q1'), 
    ('BTC-USD', 'Bitcoin (BTC)', 'Q1')
]

Q2_ASSETS = [
    ('XLE', 'Energy Sector ETF', 'Q2'), 
    ('DBC', 'Broad Commodities ETF', 'Q2')
]

Q3_ASSETS = [
    ('GLD', 'Gold ETF', 'Q3'), 
    ('LIT', 'Lithium & Battery Tech ETF', 'Q3')
]

Q4_ASSETS = [
    ('TLT', '20+ Year Treasury Bonds', 'Q4'), 
    ('XLU', 'Utilities Sector ETF', 'Q4'),
    ('UUP', 'US Dollar Index ETF', 'Q4'), 
    ('VIXY', 'Short-Term VIX Futures ETF', 'Q4')
]

# ================================================================================================
# QUADRANT DESCRIPTIONS
# ================================================================================================

QUADRANT_DESCRIPTIONS = {
    'Q1': 'Growth UP, Inflation DOWN (Goldilocks)',
    'Q2': 'Growth UP, Inflation UP (Reflation)', 
    'Q3': 'Growth DOWN, Inflation UP (Stagflation)',
    'Q4': 'Growth DOWN, Inflation DOWN (Deflation)'
}

# ================================================================================================
# API CONFIGURATION
# ================================================================================================

COINGECKO_URL = "https://api.coingecko.com/api/v3"
BINANCE_URL = "https://fapi.binance.com/fapi/v1"

# ================================================================================================
# ANALYSIS PARAMETERS
# ================================================================================================

DEFAULT_LOOKBACK_DAYS = 21
DEFAULT_DATA_DAYS_BACK = 200
DEFAULT_TOP_N_TOKENS = 8

# ================================================================================================
# STRATEGY PARAMETERS
# ================================================================================================

STRATEGY_NAME = "Quadrant Strategy"
BENCHMARK_NAME = "Buy & Hold"
RISK_FREE_RATE = 0.02  # 2%
EMA_PERIOD = 50

# ================================================================================================
# STREAMLIT CONFIGURATION
# ================================================================================================

PAGE_TITLE = "Crypto Macro Flow Dashboard"
PAGE_ICON = "📊"
LAYOUT = "wide"
REFRESH_RATE = 300  # 5 minutes in seconds
