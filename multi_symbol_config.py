"""
Multi-Symbol Trading Configuration
Enhanced support for multiple symbols with appropriate settings for each
"""

class MultiSymbolConfig:
    """Configuration for multiple symbol trading with individual parameters"""
    
    def __init__(self):
        self.symbol_configs = {
            # GOLD - Higher risk, wider stops, smaller lots
            'XAUUSD': {
                'risk_percent': 0.3,
                'base_lot': 0.01,
                'max_lot': 0.5,
                'min_stop_points': 300,
                'sl_points': 500,
                'tp_ratio': 1.5,
                'confidence_threshold': 0.7,
                'max_positions': 2,
                'description': 'Gold - High volatility, conservative approach'
            },
            
            # MAJOR PAIRS - Standard configuration
            'EURUSD': {
                'risk_percent': 0.5,
                'base_lot': 0.1,
                'max_lot': 2.0,
                'min_stop_points': 50,
                'sl_points': 100,
                'tp_ratio': 2.0,
                'confidence_threshold': 0.6,
                'max_positions': 3,
                'description': 'Euro/USD - Standard major pair'
            },
            
            'GBPUSD': {
                'risk_percent': 0.5,
                'base_lot': 0.1,
                'max_lot': 2.0,
                'min_stop_points': 50,
                'sl_points': 120,
                'tp_ratio': 1.8,
                'confidence_threshold': 0.6,
                'max_positions': 3,
                'description': 'Pound/USD - Slightly more volatile'
            },
            
            'USDJPY': {
                'risk_percent': 0.4,
                'base_lot': 0.1,
                'max_lot': 1.5,
                'min_stop_points': 50,
                'sl_points': 100,
                'tp_ratio': 2.0,
                'confidence_threshold': 0.6,
                'max_positions': 3,
                'description': 'USD/Yen - Standard major pair'
            },
            
            # MINOR PAIRS - Moderate risk
            'AUDUSD': {
                'risk_percent': 0.4,
                'base_lot': 0.1,
                'max_lot': 1.5,
                'min_stop_points': 50,
                'sl_points': 100,
                'tp_ratio': 2.0,
                'confidence_threshold': 0.65,
                'max_positions': 2,
                'description': 'Australian Dollar/USD'
            },
            
            'USDCAD': {
                'risk_percent': 0.4,
                'base_lot': 0.1,
                'max_lot': 1.5,
                'min_stop_points': 50,
                'sl_points': 100,
                'tp_ratio': 2.0,
                'confidence_threshold': 0.65,
                'max_positions': 2,
                'description': 'USD/Canadian Dollar'
            },
            
            'USDCHF': {
                'risk_percent': 0.4,
                'base_lot': 0.1,
                'max_lot': 1.5,
                'min_stop_points': 50,
                'sl_points': 100,
                'tp_ratio': 2.0,
                'confidence_threshold': 0.65,
                'max_positions': 2,      
                'description': 'USD/Swiss Franc'
            }
        }
        
        # Default configuration for unknown symbols
        self.default_config = {
            'risk_percent': 0.3,
            'base_lot': 0.01,
            'max_lot': 0.5,
            'min_stop_points': 100,
            'sl_points': 150,
            'tp_ratio': 1.5,
            'confidence_threshold': 0.7,
            'max_positions': 1,
            'description': 'Unknown symbol - Conservative approach'
        }
    
    def get_symbol_config(self, symbol):
        """Get configuration for a specific symbol"""
        # Clean symbol name
        base_symbol = symbol.replace('m', '').replace('.c', '').replace('.', '').upper()
        return self.symbol_configs.get(base_symbol, self.default_config)
    
    def get_all_configured_symbols(self):
        """Get list of all configured symbols"""
        return list(self.symbol_configs.keys())
    
    def print_symbol_config(self, symbol):
        """Print configuration for a specific symbol"""
        config = self.get_symbol_config(symbol)
        base_symbol = symbol.replace('m', '').replace('.c', '').replace('.', '').upper()
        
        print(f"\n📊 Configuration for {base_symbol}:")
        print(f"   Description: {config['description']}")
        print(f"   Risk: {config['risk_percent']}%")
        print(f"   Lot Range: {config['base_lot']} - {config['max_lot']}")
        print(f"   Stop Loss: {config['sl_points']} points")
        print(f"   Take Profit Ratio: {config['tp_ratio']}:1")
        print(f"   Confidence Threshold: {config['confidence_threshold']}")
        print(f"   Max Positions: {config['max_positions']}")
    
    def print_all_configs(self):
        """Print all symbol configurations"""
        print("\n🌍 Multi-Symbol Trading Configuration:")
        print("=" * 60)
        
        for symbol in self.symbol_configs:
            self.print_symbol_config(symbol)
        
        print(f"\n🔧 Default Config (Unknown Symbols):")
        print(f"   Risk: {self.default_config['risk_percent']}%")
        print(f"   Lot Range: {self.default_config['base_lot']} - {self.default_config['max_lot']}")
        print(f"   Stop Loss: {self.default_config['sl_points']} points")
    
    def validate_symbol_support(self, symbols):
        """Check which symbols are configured and which use defaults"""
        configured = []
        using_defaults = []
        
        for symbol in symbols:
            base_symbol = symbol.replace('m', '').replace('.c', '').replace('.', '').upper()
            if base_symbol in self.symbol_configs:
                configured.append(base_symbol)
            else:
                using_defaults.append(base_symbol)
        
        print(f"\n✅ Configured Symbols: {', '.join(configured) if configured else 'None'}")
        print(f"⚠️  Default Config Used: {', '.join(using_defaults) if using_defaults else 'None'}")
        
        return configured, using_defaults

# Global instance
multi_symbol_config = MultiSymbolConfig()

if __name__ == "__main__":
    # Test the configuration system
    config = MultiSymbolConfig()
    config.print_all_configs()
    
    # Test symbol validation
    test_symbols = ['XAUUSD', 'EURUSD', 'BTCUSD', 'GBPUSDm']
    config.validate_symbol_support(test_symbols)
