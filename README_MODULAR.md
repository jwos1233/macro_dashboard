# Macro Dashboard - Modular Version

A comprehensive, modular crypto macro flow dashboard built with Streamlit, designed for market regime analysis and strategy performance tracking.

## 🏗️ Modular Architecture

The dashboard has been restructured into logical, maintainable modules:

### Core Modules

- **`config.py`** - Centralized configuration and constants
- **`data_fetcher.py`** - Data retrieval and processing utilities
- **`quadrant_analysis.py`** - Market regime classification logic
- **`axe_list_generator.py`** - Crypto token analysis and ranking
- **`strategy_performance.py`** - Strategy backtesting and metrics
- **`main_dashboard.py`** - Main Streamlit interface and integration

### Benefits of Modular Structure

✅ **Maintainability** - Each module has a single responsibility  
✅ **Reusability** - Modules can be imported and used independently  
✅ **Testing** - Individual modules can be tested in isolation  
✅ **Scalability** - Easy to add new features or modify existing ones  
✅ **Collaboration** - Multiple developers can work on different modules  

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
python run_dashboard.py
```

Or directly with Streamlit:
```bash
streamlit run main_dashboard.py
```

## 📊 Features

### Market Quadrant Analysis
- **Real-time Classification** - Current market regime (Q1-Q4)
- **Historical Tracking** - 90-day quadrant history with color coding
- **Confidence Scoring** - Regime strength and confidence metrics
- **Asset Mapping** - Core assets classified by quadrant

### Strategy Performance
- **Backtesting Engine** - Quadrant-based strategy vs buy & hold
- **Performance Metrics** - Sharpe ratio, drawdown, win rate
- **Portfolio Analysis** - 50/50 BTC+ETH portfolio performance
- **Visual Charts** - Interactive performance comparison

### Axe List Generation
- **Token Ranking** - Top performing crypto tokens by momentum
- **Performance Metrics** - 1-day, 7-day, and 30-day returns
- **Market Data** - Real-time data from CoinGecko and Binance
- **Configurable Limits** - Adjustable number of top tokens

## 🔧 Configuration

All configuration is centralized in `config.py`:

```python
# Analysis parameters
DEFAULT_LOOKBACK_DAYS = 21
DEFAULT_DATA_DAYS_BACK = 200
DEFAULT_TOP_N_TOKENS = 8

# Strategy parameters
STRATEGY_NAME = "Quadrant Strategy"
BENCHMARK_NAME = "Buy & Hold"
RISK_FREE_RATE = 0.02  # 2%
EMA_PERIOD = 50
```

## 📁 File Structure

```
macro_dashboard/
├── config.py                 # Configuration and constants
├── data_fetcher.py          # Data retrieval utilities
├── quadrant_analysis.py     # Market regime analysis
├── axe_list_generator.py    # Token ranking and analysis
├── strategy_performance.py  # Strategy backtesting
├── main_dashboard.py        # Main Streamlit interface
├── run_dashboard.py         # Launcher script
├── requirements.txt         # Python dependencies
└── README_MODULAR.md        # This file
```

## 🧪 Testing Individual Modules

Each module can be tested independently:

```python
# Test quadrant analysis
from quadrant_analysis import CurrentQuadrantAnalysis
analyzer = CurrentQuadrantAnalysis()
results = analyzer.analyze_current_quadrant_and_30_days()

# Test data fetching
from data_fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.fetch_recent_data(['BTC-USD', 'ETH-USD'])

# Test strategy performance
from strategy_performance import StrategyPerformanceAnalysis
analyzer = StrategyPerformanceAnalysis()
performance = analyzer.calculate_strategy_performance(price_data, results)
```

## 🔄 Data Flow

1. **Data Fetching** → `data_fetcher.py` retrieves market data
2. **Quadrant Analysis** → `quadrant_analysis.py` classifies market regime
3. **Strategy Performance** → `strategy_performance.py` calculates metrics
4. **Axe List** → `axe_list_generator.py` ranks crypto tokens
5. **Integration** → `main_dashboard.py` combines all modules
6. **Display** → Streamlit renders the interactive interface

## 🚀 Deployment

### Local Development
```bash
cd macro_dashboard
python run_dashboard.py
```

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Set main file to `main_dashboard.py`
4. Deploy automatically

### Docker (Future Enhancement)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main_dashboard.py"]
```

## 🔧 Customization

### Adding New Assets
Edit `config.py` to add new assets to quadrants:

```python
Q1_ASSETS = [
    ('QQQ', 'NASDAQ 100 (Growth)', 'Q1'),
    ('NEW_ASSET', 'New Asset Name', 'Q1')  # Add here
]
```

### Modifying Strategy Logic
Edit `strategy_performance.py` to change strategy rules:

```python
# Example: Change EMA period
crypto_50ema = crypto_prices.rolling(window=20).mean()  # 20-day EMA
```

### Adding New Data Sources
Extend `data_fetcher.py` with new methods:

```python
def fetch_alternative_data(self, symbol: str):
    # Implement new data source
    pass
```

## 📈 Performance Monitoring

The dashboard includes built-in performance monitoring:

- **Auto-refresh** every 5 minutes
- **Error handling** with user-friendly messages
- **Loading states** for long-running operations
- **Session state** management for data persistence

## 🤝 Contributing

When adding new features:

1. **Create new module** if adding major functionality
2. **Extend existing module** for minor changes
3. **Update config.py** for new constants
4. **Add tests** for new functionality
5. **Update documentation** in this README

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Data Fetching Issues**: Check internet connection and API limits
- Yahoo Finance has rate limits
- CoinGecko requires API key for high volume

**Chart Display Issues**: Verify Plotly installation
```bash
pip install plotly
```

### Debug Mode

Enable debug output by modifying `main_dashboard.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Future Enhancements

- **Database Integration** - Persistent storage for historical data
- **Real-time Updates** - WebSocket connections for live data
- **Advanced Analytics** - Machine learning models for regime prediction
- **Portfolio Management** - Integration with trading platforms
- **Mobile App** - React Native companion app

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Streamlit, Pandas, and Plotly**
