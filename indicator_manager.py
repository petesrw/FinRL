# 🤖 Automatic Indicator Selection System
# Implementation of smart_defaults, adaptive, and meta_learning methods

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime, timedelta

class OptimizationMethod(Enum):
    SMART_DEFAULTS = "smart_defaults"
    ADAPTIVE = "adaptive"
    META_LEARNING = "meta_learning"

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    sma_fast: int = 20
    sma_slow: int = 50
    atr_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    ema_fast: int = 12
    ema_slow: int = 26
    williams_r_period: int = 14
    cci_period: int = 20
    
    def to_dict(self) -> Dict:
        return {
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'sma_fast': self.sma_fast,
            'sma_slow': self.sma_slow,
            'atr_period': self.atr_period,
            'stoch_k': self.stoch_k,
            'stoch_d': self.stoch_d,
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'williams_r_period': self.williams_r_period,
            'cci_period': self.cci_period
        }

@dataclass
class MarketConditions:
    """Current market conditions for adaptive optimization"""
    volatility: float = 0.0
    trend_strength: float = 0.0
    volume_trend: float = 0.0
    price_momentum: float = 0.0
    market_regime: str = "ranging"  # trending, ranging, volatile

class SmartIndicatorManager:
    """
    🚀 Automatic Indicator Selection System
    
    Supports three optimization methods:
    1. smart_defaults - Currency pair specific defaults
    2. adaptive - Performance-based optimization
    3. meta_learning - RL-based indicator selection
    """
    
    def __init__(self, 
                 symbol: str, 
                 timeframe: str = "H1",
                 optimization_method: str = "smart_defaults",
                 auto_select: bool = True):
        
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.optimization_method = OptimizationMethod(optimization_method)
        self.auto_select = auto_select
        
        # Performance tracking
        self.performance_history = []
        self.config_history = []
        self.last_optimization = datetime.now()
        
        # Adaptive parameters
        self.performance_threshold = 0.65
        self.optimization_period = 1000  # steps
        self.min_data_points = 100
        
        # Meta-learning components (initialize before config)
        self.indicator_library = self._build_indicator_library()
        self.selection_weights = self._initialize_selection_weights()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Current configuration (initialize after meta-learning components)
        self.current_config = self._initialize_config()
        
    def _initialize_config(self) -> IndicatorConfig:
        """Initialize configuration based on optimization method"""
        if not self.auto_select:
            return self._get_manual_config()
            
        if self.optimization_method == OptimizationMethod.SMART_DEFAULTS:
            return self._get_smart_defaults()
        elif self.optimization_method == OptimizationMethod.ADAPTIVE:
            return self._get_smart_defaults()  # Start with smart defaults
        else:  # META_LEARNING
            return self._get_meta_learning_config()
    
    def _get_manual_config(self) -> IndicatorConfig:
        """Get manual configuration from environment variables"""
        from config import Config
        config = Config()
        
        return IndicatorConfig(
            rsi_period=getattr(config, 'RSI_PERIOD', 14),
            macd_fast=getattr(config, 'MACD_FAST', 12),
            macd_slow=getattr(config, 'MACD_SLOW', 26),
            bb_period=getattr(config, 'BB_PERIOD', 20),
            sma_fast=getattr(config, 'SMA_FAST', 20),
            sma_slow=getattr(config, 'SMA_SLOW', 50),
            atr_period=getattr(config, 'ATR_PERIOD', 14)
        )
    
    def _get_smart_defaults(self) -> IndicatorConfig:
        """
        🎯 Smart Defaults: Currency pair specific configurations
        Based on 100+ years of trading knowledge
        """
        
        # Major pairs (EUR/USD, GBP/USD, USD/CHF)
        if self.symbol in ["EURUSD", "GBPUSD", "USDCHF"]:
            return IndicatorConfig(
                rsi_period=14,
                macd_fast=12, macd_slow=26, macd_signal=9,
                bb_period=20, bb_std=2.0,
                sma_fast=20, sma_slow=50,
                atr_period=14,
                stoch_k=14, stoch_d=3,
                ema_fast=12, ema_slow=26,
                williams_r_period=14,
                cci_period=20
            )
        
        # Yen pairs (USD/JPY, EUR/JPY, GBP/JPY)
        elif "JPY" in self.symbol:
            return IndicatorConfig(
                rsi_period=21,      # Yen moves differently
                macd_fast=8, macd_slow=21, macd_signal=7,
                bb_period=15, bb_std=1.8,
                sma_fast=15, sma_slow=45,
                atr_period=10,
                stoch_k=10, stoch_d=3,
                ema_fast=10, ema_slow=24,
                williams_r_period=12,
                cci_period=15
            )
        
        # Commodity currencies (AUD, CAD, NZD)
        elif any(curr in self.symbol for curr in ["AUD", "CAD", "NZD"]):
            return IndicatorConfig(
                rsi_period=18,
                macd_fast=10, macd_slow=24, macd_signal=8,
                bb_period=18, bb_std=2.2,
                sma_fast=18, sma_slow=48,
                atr_period=12,
                stoch_k=12, stoch_d=3,
                ema_fast=11, ema_slow=25,
                williams_r_period=16,
                cci_period=18
            )
        
        # Gold (XAUUSD) - Special configuration for precious metals
        elif "XAU" in self.symbol or "GOLD" in self.symbol:
            return IndicatorConfig(
                rsi_period=14,      # Standard RSI for gold
                macd_fast=12, macd_slow=26, macd_signal=9,  # Standard MACD
                bb_period=20, bb_std=2.5,   # Wider bands for gold volatility
                sma_fast=20, sma_slow=50,   # Standard MAs
                atr_period=14,      # Standard ATR
                stoch_k=14, stoch_d=3,      # Standard Stochastic
                ema_fast=12, ema_slow=26,   # Standard EMAs
                williams_r_period=14,       # Standard Williams %R
                cci_period=20       # Standard CCI
            )
        
        # Exotic pairs
        elif any(curr in self.symbol for curr in ["ZAR", "TRY", "MXN", "PLN"]):
            return IndicatorConfig(
                rsi_period=10,      # More sensitive for volatile pairs
                macd_fast=6, macd_slow=18, macd_signal=6,
                bb_period=12, bb_std=2.5,
                sma_fast=12, sma_slow=36,
                atr_period=8,
                stoch_k=8, stoch_d=2,
                ema_fast=8, ema_slow=20,
                williams_r_period=10,
                cci_period=12
            )
        
        # Default for other pairs
        else:
            return IndicatorConfig()  # Standard defaults
    
    def _build_indicator_library(self) -> Dict[str, List[str]]:
        """Build library of available indicators by category"""
        return {
            "trend": ["sma", "ema", "macd", "adx", "parabolic_sar"],
            "momentum": ["rsi", "stochastic", "williams_r", "cci", "momentum"],
            "volatility": ["bollinger", "atr", "keltner", "donchian"],
            "volume": ["obv", "mfi", "vwap", "ad_line"],
            "support_resistance": ["pivot_points", "fibonacci", "support_resistance"]
        }
    
    def _initialize_selection_weights(self) -> Dict[str, float]:
        """Initialize weights for meta-learning indicator selection"""
        weights = {}
        for category, indicators in self.indicator_library.items():
            for indicator in indicators:
                weights[indicator] = np.random.uniform(0.3, 0.7)
        return weights
    
    def _get_meta_learning_config(self) -> IndicatorConfig:
        """
        🧠 Meta-Learning: RL-based indicator selection
        Agent learns which indicators work best
        """
        # Start with base config and modify based on learned weights
        config = IndicatorConfig()
        
        # Adjust parameters based on learned weights
        if self.selection_weights.get("rsi", 0.5) > 0.6:
            config.rsi_period = int(14 * (1 + (self.selection_weights["rsi"] - 0.5)))
        
        if self.selection_weights.get("macd", 0.5) > 0.6:
            config.macd_fast = max(6, int(12 * self.selection_weights["macd"]))
            config.macd_slow = max(12, int(26 * self.selection_weights["macd"]))
        
        if self.selection_weights.get("bollinger", 0.5) > 0.6:
            config.bb_period = max(10, int(20 * self.selection_weights["bollinger"]))
        
        return config
    
    def calculate_market_conditions(self, data: pd.DataFrame) -> MarketConditions:
        """
        📊 Calculate current market conditions for adaptive optimization
        """
        if len(data) < 50:
            return MarketConditions()
        
        # Calculate volatility (ATR-based)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean()
        volatility = (atr.iloc[-1] / data['close'].iloc[-1]) * 100
        
        # Calculate trend strength (ADX-like)
        price_change = data['close'].diff()
        up_moves = price_change.where(price_change > 0, 0)
        down_moves = -price_change.where(price_change < 0, 0)
        
        up_sma = up_moves.rolling(14).mean()
        down_sma = down_moves.rolling(14).mean()
        
        rs = up_sma / down_sma
        trend_strength = abs(100 - (100 / (1 + rs.iloc[-1])) - 50) / 50
        
        # Calculate price momentum
        momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
        
        # Determine market regime
        if volatility > 2.0:
            regime = "volatile"
        elif abs(momentum) > 3.0:
            regime = "trending"
        else:
            regime = "ranging"
        
        return MarketConditions(
            volatility=min(volatility, 10.0),  # Cap at 10%
            trend_strength=min(trend_strength, 1.0),
            price_momentum=momentum,
            market_regime=regime
        )
    
    def adaptive_optimization(self, 
                            market_data: pd.DataFrame, 
                            performance_score: float) -> IndicatorConfig:
        """
        🔧 Adaptive Optimization: Adjust indicators based on performance
        """
        conditions = self.calculate_market_conditions(market_data)
        config = self.current_config
        
        # Track performance
        self.performance_history.append(performance_score)
        self.config_history.append(config.to_dict())
        
        # Only optimize if we have enough data and performance is below threshold
        if (len(self.performance_history) < self.min_data_points or 
            performance_score > self.performance_threshold):
            return config
        
        # Calculate recent performance trend
        recent_performance = np.mean(self.performance_history[-20:])
        
        if recent_performance < self.performance_threshold:
            self.logger.info(f"Performance below threshold ({recent_performance:.3f}), optimizing...")
            
            # Adapt to market conditions
            new_config = IndicatorConfig(**config.to_dict())
            
            # High volatility adjustments
            if conditions.volatility > 3.0:
                new_config.rsi_period = max(8, int(config.rsi_period * 0.7))
                new_config.atr_period = max(6, int(config.atr_period * 0.8))
                new_config.bb_period = max(10, int(config.bb_period * 0.8))
                self.logger.info("Adjusted for high volatility")
            
            # Strong trend adjustments
            if conditions.trend_strength > 0.7:
                new_config.sma_fast = max(10, int(config.sma_fast * 0.8))
                new_config.sma_slow = max(20, int(config.sma_slow * 0.9))
                new_config.macd_fast = max(6, int(config.macd_fast * 0.8))
                self.logger.info("Adjusted for strong trend")
            
            # Ranging market adjustments
            if conditions.market_regime == "ranging":
                new_config.rsi_period = min(28, int(config.rsi_period * 1.3))
                new_config.bb_std = min(3.0, config.bb_std * 1.2)
                self.logger.info("Adjusted for ranging market")
            
            return new_config
        
        return config
    
    def meta_learning_update(self, 
                           indicator_performance: Dict[str, float],
                           market_conditions: MarketConditions):
        """
        🧠 Update meta-learning weights based on indicator performance
        """
        learning_rate = 0.01
        
        for indicator, performance in indicator_performance.items():
            if indicator in self.selection_weights:
                # Update weight based on performance
                current_weight = self.selection_weights[indicator]
                
                # Reward good performance, penalize bad performance
                if performance > 0.6:
                    self.selection_weights[indicator] = min(1.0, 
                        current_weight + learning_rate * (performance - 0.5))
                else:
                    self.selection_weights[indicator] = max(0.1, 
                        current_weight - learning_rate * (0.5 - performance))
        
        # Normalize weights
        total_weight = sum(self.selection_weights.values())
        for indicator in self.selection_weights:
            self.selection_weights[indicator] /= total_weight
    
    def get_optimal_config(self, 
                          market_data: Optional[pd.DataFrame] = None,
                          performance_score: Optional[float] = None) -> IndicatorConfig:
        """
        🎯 Get optimal indicator configuration based on selected method
        """
        
        if not self.auto_select:
            return self._get_manual_config()
        
        if self.optimization_method == OptimizationMethod.SMART_DEFAULTS:
            return self._get_smart_defaults()
        
        elif self.optimization_method == OptimizationMethod.ADAPTIVE:
            if market_data is not None and performance_score is not None:
                self.current_config = self.adaptive_optimization(market_data, performance_score)
            return self.current_config
        
        else:  # META_LEARNING
            if market_data is not None:
                conditions = self.calculate_market_conditions(market_data)
                # Update config based on current market conditions and learned weights
                self.current_config = self._get_meta_learning_config()
            return self.current_config
    
    def should_reoptimize(self) -> bool:
        """Check if it's time to reoptimize configuration"""
        if self.optimization_method == OptimizationMethod.SMART_DEFAULTS:
            return False
        
        time_since_last = datetime.now() - self.last_optimization
        return (time_since_last.total_seconds() > 3600 or  # 1 hour
                len(self.performance_history) % self.optimization_period == 0)
    
    def save_config(self, filepath: str = "indicator_config.json"):
        """Save current configuration to file"""
        config_data = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "optimization_method": self.optimization_method.value,
            "current_config": self.current_config.to_dict(),
            "selection_weights": self.selection_weights,
            "performance_history": self.performance_history[-100:],  # Last 100 only
            "last_update": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, filepath: str = "indicator_config.json"):
        """Load configuration from file"""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            if config_data.get("symbol") == self.symbol:
                self.selection_weights = config_data.get("selection_weights", {})
                self.performance_history = config_data.get("performance_history", [])
                
                # Restore current config
                config_dict = config_data.get("current_config", {})
                self.current_config = IndicatorConfig(**config_dict)
                
                self.logger.info(f"Loaded configuration for {self.symbol}")
        
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
    
    def get_config_summary(self) -> Dict:
        """Get summary of current configuration"""
        return {
            "symbol": self.symbol,
            "optimization_method": self.optimization_method.value,
            "auto_select": self.auto_select,
            "current_config": self.current_config.to_dict(),
            "performance_avg": np.mean(self.performance_history[-20:]) if self.performance_history else 0.0,
            "total_optimizations": len(self.config_history),
            "last_optimization": self.last_optimization.isoformat()
        }

# 🚀 Factory function for easy creation
def create_indicator_manager(symbol: str, 
                           optimization_method: str = "smart_defaults",
                           auto_select: bool = True,
                           timeframe: str = "H1") -> SmartIndicatorManager:
    """
    Factory function to create indicator manager with proper configuration
    
    Args:
        symbol: Currency pair (e.g., "EURUSD")
        optimization_method: "smart_defaults", "adaptive", or "meta_learning"
        auto_select: Enable automatic selection
        timeframe: Trading timeframe
    
    Returns:
        Configured SmartIndicatorManager instance
    """
    
    manager = SmartIndicatorManager(
        symbol=symbol,
        timeframe=timeframe,
        optimization_method=optimization_method,
        auto_select=auto_select
    )
    
    # Try to load existing configuration
    config_file = f"indicator_config_{symbol}_{optimization_method}.json"
    manager.load_config(config_file)
    
    return manager

if __name__ == "__main__":
    # 🧪 Example usage
    
    # Test smart defaults
    print("🎯 Testing Smart Defaults:")
    manager_defaults = create_indicator_manager("EURUSD", "smart_defaults")
    config = manager_defaults.get_optimal_config()
    print(f"EUR/USD config: {config.to_dict()}")
    
    # Test different currency pairs
    for symbol in ["USDJPY", "AUDUSD", "USDTRY"]:
        manager = create_indicator_manager(symbol, "smart_defaults")
        config = manager.get_optimal_config()
        print(f"{symbol} config: RSI={config.rsi_period}, MACD=({config.macd_fast},{config.macd_slow})")
    
    print("\n🔧 Testing Adaptive Optimization:")
    manager_adaptive = create_indicator_manager("EURUSD", "adaptive")
    
    # Simulate market data and performance
    sample_data = pd.DataFrame({
        'open': np.random.randn(100) + 1.1000,
        'high': np.random.randn(100) + 1.1010,
        'low': np.random.randn(100) + 1.0990,
        'close': np.random.randn(100) + 1.1000,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Test adaptive optimization
    config = manager_adaptive.get_optimal_config(sample_data, 0.45)  # Low performance
    print(f"Adaptive config after low performance: {config.to_dict()}")
    
    print("\n🧠 Testing Meta-Learning:")
    manager_meta = create_indicator_manager("EURUSD", "meta_learning")
    config = manager_meta.get_optimal_config(sample_data)
    print(f"Meta-learning config: {config.to_dict()}")
    
    print("\n📊 Configuration Summary:")
    summary = manager_adaptive.get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")