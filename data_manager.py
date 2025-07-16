#!/usr/bin/env python3
"""
📊 Real Market Data Manager
Downloads and manages real historical forex data for training
"""

import pandas as pd
import numpy as np
import yfinance as yf
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import logging
import os
import requests
import time

class MarketDataManager:
    """
    📊 Manages real market data from multiple sources
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        
    def get_yahoo_data(self, symbol, period="2y", interval="1h"):
        """
        📈 Get data from Yahoo Finance
        
        Args:
            symbol: Forex symbol (e.g., "EURUSD=X")
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        
        # Convert symbol to Yahoo Finance format
        yahoo_symbols = {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X", 
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCHF": "USDCHF=X",
            "USDCAD": "USDCAD=X",
            "NZDUSD": "NZDUSD=X",
            "XAUUSD": "GC=F",  # Gold futures
            "XAGUSD": "SI=F",  # Silver futures
        }
        
        yahoo_symbol = yahoo_symbols.get(symbol, f"{symbol}=X")
        
        try:
            self.logger.info(f"Downloading {symbol} data from Yahoo Finance...")
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.error(f"No data received for {symbol}")
                return None
            
            # Convert to our format
            forex_data = pd.DataFrame({
                'time': data.index,
                'open': data['Open'].values,
                'high': data['High'].values,
                'low': data['Low'].values,
                'close': data['Close'].values,
                'volume': data['Volume'].values if 'Volume' in data.columns else np.ones(len(data)) * 1000
            })
            
            # Remove NaN values
            forex_data = forex_data.dropna()
            
            self.logger.info(f"Downloaded {len(forex_data)} data points for {symbol}")
            return forex_data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance download failed for {symbol}: {e}")
            return None
    
    def get_mt5_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, count=10000):
        """
        📊 Get data from MetaTrader 5
        
        Args:
            symbol: MT5 symbol (e.g., "EURUSD")
            timeframe: MT5 timeframe constant
            count: Number of bars to download
        """
        
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return None
            
            self.logger.info(f"Downloading {symbol} data from MT5...")
            
            # Get data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No MT5 data received for {symbol}")
                return None
            
            # Convert to DataFrame
            forex_data = pd.DataFrame(rates)
            forex_data['time'] = pd.to_datetime(forex_data['time'], unit='s')
            
            # Rename columns to match our format
            forex_data = forex_data.rename(columns={
                'tick_volume': 'volume'
            })
            
            # Select required columns
            forex_data = forex_data[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            self.logger.info(f"Downloaded {len(forex_data)} data points for {symbol} from MT5")
            return forex_data
            
        except Exception as e:
            self.logger.error(f"MT5 download failed for {symbol}: {e}")
            return None
        finally:
            mt5.shutdown()
    
    def get_alpha_vantage_data(self, symbol, api_key, function="FX_INTRADAY", interval="60min"):
        """
        📈 Get data from Alpha Vantage (requires API key)
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            api_key: Alpha Vantage API key
            function: API function (FX_INTRADAY, FX_DAILY, etc.)
            interval: Data interval (1min, 5min, 15min, 30min, 60min)
        """
        
        if not api_key:
            self.logger.warning("Alpha Vantage API key not provided")
            return None
        
        try:
            from_symbol = symbol[:3]
            to_symbol = symbol[3:]
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'interval': interval,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            self.logger.info(f"Downloading {symbol} data from Alpha Vantage...")
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage note: {data['Note']}")
                return None
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                self.logger.error("No time series data found in Alpha Vantage response")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'time': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 1000  # Alpha Vantage doesn't provide forex volume
                })
            
            forex_data = pd.DataFrame(df_data)
            forex_data = forex_data.sort_values('time').reset_index(drop=True)
            
            self.logger.info(f"Downloaded {len(forex_data)} data points for {symbol} from Alpha Vantage")
            return forex_data
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage download failed for {symbol}: {e}")
            return None
    
    def get_real_data(self, symbol, source="auto", **kwargs):
        """
        🎯 Get real market data from the best available source
        
        Args:
            symbol: Currency symbol (e.g., "EURUSD")
            source: Data source ("auto", "yahoo", "mt5", "alpha_vantage")
            **kwargs: Additional parameters for specific sources
        """
        
        cache_key = f"{symbol}_{source}_{str(kwargs)}"
        
        # Check cache first
        if cache_key in self.data_cache:
            self.logger.info(f"Using cached data for {symbol}")
            return self.data_cache[cache_key]
        
        data = None
        
        if source == "auto":
            # Try sources in order of preference
            sources_to_try = ["yahoo", "mt5", "alpha_vantage"]
            
            for src in sources_to_try:
                self.logger.info(f"Trying {src} for {symbol}...")
                
                if src == "yahoo":
                    data = self.get_yahoo_data(symbol, **kwargs)
                elif src == "mt5":
                    data = self.get_mt5_data(symbol, **kwargs)
                elif src == "alpha_vantage":
                    api_key = kwargs.get("api_key")
                    if api_key:
                        data = self.get_alpha_vantage_data(symbol, api_key, **kwargs)
                
                if data is not None and len(data) > 100:
                    self.logger.info(f"Successfully got data from {src}")
                    break
                else:
                    self.logger.warning(f"{src} failed or insufficient data")
        
        elif source == "yahoo":
            data = self.get_yahoo_data(symbol, **kwargs)
        elif source == "mt5":
            data = self.get_mt5_data(symbol, **kwargs)
        elif source == "alpha_vantage":
            data = self.get_alpha_vantage_data(symbol, **kwargs)
        
        # Cache successful results
        if data is not None and len(data) > 100:
            self.data_cache[cache_key] = data
            
            # Save to file for future use
            filename = f"data_{symbol}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
            data.to_csv(filename, index=False)
            self.logger.info(f"Saved data to {filename}")
        
        return data
    
    def load_saved_data(self, symbol, max_age_days=7):
        """
        📁 Load previously saved data if recent enough
        """
        
        # Look for recent data files
        pattern = f"data_{symbol}_"
        recent_files = []
        
        for filename in os.listdir('.'):
            if filename.startswith(pattern) and filename.endswith('.csv'):
                try:
                    # Extract date from filename
                    date_str = filename.split('_')[-1].replace('.csv', '')
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    # Check if file is recent enough
                    if (datetime.now() - file_date).days <= max_age_days:
                        recent_files.append((filename, file_date))
                except:
                    continue
        
        if recent_files:
            # Use most recent file
            recent_files.sort(key=lambda x: x[1], reverse=True)
            filename = recent_files[0][0]
            
            try:
                data = pd.read_csv(filename)
                data['time'] = pd.to_datetime(data['time'])
                self.logger.info(f"Loaded saved data from {filename}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load {filename}: {e}")
        
        return None
    
    def get_training_data(self, symbol, prefer_saved=True, **kwargs):
        """
        🎯 Get training data with smart caching
        
        Args:
            symbol: Currency symbol
            prefer_saved: Try to use saved data first
            **kwargs: Parameters for data download
        """
        
        data = None
        
        # Try saved data first if preferred
        if prefer_saved:
            data = self.load_saved_data(symbol)
            if data is not None:
                return data
        
        # Download fresh data
        self.logger.info(f"Downloading fresh data for {symbol}...")
        data = self.get_real_data(symbol, **kwargs)
        
        if data is None:
            self.logger.error(f"Failed to get real data for {symbol}, falling back to simulated data")
            return None
        
        return data

# 🚀 Enhanced Environment with Real Data
class RealDataForexEnvironment:
    """
    Enhanced Forex Environment using real market data
    """
    
    def __init__(self, symbol="EURUSD", use_real_data=True, **kwargs):
        self.symbol = symbol
        self.use_real_data = use_real_data
        self.data_manager = MarketDataManager()
        self.logger = logging.getLogger(__name__)
        
        # Load data
        if use_real_data:
            self.data = self.load_real_data(**kwargs)
        else:
            self.data = self.generate_simulated_data()
        
        if self.data is None or len(self.data) < 1000:
            self.logger.warning("Insufficient real data, falling back to simulated data")
            self.data = self.generate_simulated_data()
    
    def load_real_data(self, **kwargs):
        """Load real market data"""
        self.logger.info(f"Loading real market data for {self.symbol}...")
        
        # Try to get real data
        data = self.data_manager.get_training_data(
            self.symbol, 
            prefer_saved=True,
            period="2y",  # 2 years of data
            interval="1h",  # 1-hour intervals
            **kwargs
        )
        
        if data is not None and len(data) > 1000:
            self.logger.info(f"Loaded {len(data)} real data points for {self.symbol}")
            return data
        else:
            self.logger.warning(f"Failed to load sufficient real data for {self.symbol}")
            return None
    
    def generate_simulated_data(self):
        """Fallback: Generate simulated data"""
        self.logger.info(f"Generating simulated data for {self.symbol}")
        
        # Use the existing simulation logic
        # (This would be the same as the current implementation)
        base_prices = {
            "EURUSD": 1.1000, "GBPUSD": 1.3000, "USDJPY": 110.00,
            "XAUUSD": 1800.00, "USDCHF": 0.9200, "USDCAD": 1.2500
        }
        
        base_price = base_prices.get(self.symbol, 1.1000)
        periods = 10000  # More data points
        
        # Generate realistic price movements
        np.random.seed(42)
        volatility = 0.0008 if "USD" in self.symbol else 0.015
        
        returns = []
        trend = 0.0
        
        for i in range(periods):
            if i > 0:
                trend = 0.95 * trend + 0.05 * np.random.normal(0, volatility)
            else:
                trend = np.random.normal(0, volatility)
            
            noise = np.random.normal(0, volatility)
            daily_return = trend + noise
            returns.append(daily_return)
        
        returns = np.array(returns)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        dates = pd.date_range(start='2022-01-01', periods=periods, freq='1H')
        
        data = []
        for i, price in enumerate(prices):
            volatility_factor = np.random.uniform(0.5, 1.5)
            spread = volatility * volatility_factor * price
            
            high = price + np.random.uniform(0, spread)
            low = price - np.random.uniform(0, spread)
            
            if i == 0:
                open_price = price
            else:
                open_price = data[-1]['close']
            
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            data.append({
                'time': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })
        
        return pd.DataFrame(data)

def test_data_sources():
    """🧪 Test different data sources"""
    print("🧪 Testing Market Data Sources")
    print("=" * 50)
    
    manager = MarketDataManager()
    test_symbol = "EURUSD"
    
    # Test Yahoo Finance
    print(f"\n📈 Testing Yahoo Finance for {test_symbol}...")
    yahoo_data = manager.get_yahoo_data(test_symbol, period="1mo", interval="1h")
    if yahoo_data is not None:
        print(f"✅ Yahoo: {len(yahoo_data)} data points")
        print(f"   Date range: {yahoo_data['time'].min()} to {yahoo_data['time'].max()}")
    else:
        print("❌ Yahoo Finance failed")
    
    # Test MT5 (if available)
    print(f"\n📊 Testing MT5 for {test_symbol}...")
    mt5_data = manager.get_mt5_data(test_symbol, count=1000)
    if mt5_data is not None:
        print(f"✅ MT5: {len(mt5_data)} data points")
        print(f"   Date range: {mt5_data['time'].min()} to {mt5_data['time'].max()}")
    else:
        print("❌ MT5 failed (may not be configured)")
    
    # Test automatic selection
    print(f"\n🎯 Testing automatic data source for {test_symbol}...")
    auto_data = manager.get_real_data(test_symbol, source="auto")
    if auto_data is not None:
        print(f"✅ Auto: {len(auto_data)} data points")
        print(f"   Date range: {auto_data['time'].min()} to {auto_data['time'].max()}")
    else:
        print("❌ All sources failed")

if __name__ == "__main__":
    test_data_sources()