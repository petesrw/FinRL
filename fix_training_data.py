#!/usr/bin/env python3
"""
Fix and properly format training data
"""

import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_symbol_data(symbol):
    """
    Fix data for a specific symbol
    """
    logger.info(f"Fixing {symbol} data...")
    
    # Read original file
    original_file = f"train_data/{symbol}/{symbol}_M5.csv"
    
    if not os.path.exists(original_file):
        logger.error(f"File not found: {original_file}")
        return None
    
    try:
        # Read with tab separator
        df = pd.read_csv(original_file, sep='\t')
        logger.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Handle extra columns - keep only first 6 columns
        if len(df.columns) > 6:
            df = df.iloc[:, :6]
            df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Rename columns to standard format
        df = df.rename(columns={
            'Time': 'timestamp',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Convert timestamp properly
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert volume to int
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # Remove rows with invalid data
        df = df.dropna(subset=['timestamp'] + price_cols)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate data
        if len(df) == 0:
            logger.error(f"No valid data after cleaning for {symbol}")
            return None
        
        # Check for reasonable price ranges
        for col in price_cols:
            if df[col].min() <= 0:
                logger.warning(f"Found non-positive prices in {symbol} {col}")
        
        # Save fixed data
        output_file = f"train_data/{symbol}/{symbol}_M5_clean.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved clean data to {output_file}")
        
        # Print summary
        print(f"\n📊 {symbol} Clean Data:")
        print(f"   Rows: {len(df):,}")
        if len(df) > 0:
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
        print(f"   Output: {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def main():
    """
    Fix all symbol data
    """
    print("🔧 Fixing training data format...")
    print("="*50)
    
    symbols = ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY', 'XAUUSD']
    
    results = {}
    all_data = []
    
    for symbol in symbols:
        df = fix_symbol_data(symbol)
        if df is not None and len(df) > 0:
            results[symbol] = len(df)
            df['symbol'] = symbol
            all_data.append(df)
    
    # Create combined dataset
    if all_data:
        print("\n" + "="*40)
        print("Creating combined dataset...")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Save combined data
        output_file = "train_data/all_symbols_M5_clean.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n📈 Combined Clean Dataset:")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Symbols: {', '.join(sorted(combined_df['symbol'].unique()))}")
        print(f"   Output: {output_file}")
        
        # Create summary
        summary_data = []
        for symbol in sorted(combined_df['symbol'].unique()):
            symbol_data = combined_df[combined_df['symbol'] == symbol]
            if len(symbol_data) > 0:
                summary_data.append({
                    'symbol': symbol,
                    'rows': len(symbol_data),
                    'start_date': symbol_data['timestamp'].min(),
                    'end_date': symbol_data['timestamp'].max(),
                    'min_price': symbol_data['close'].min(),
                    'max_price': symbol_data['close'].max(),
                    'avg_price': symbol_data['close'].mean()
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('train_data/clean_data_summary.csv', index=False)
            print(f"   Summary: train_data/clean_data_summary.csv")
        
        print("\n" + "="*50)
        print("✅ DATA FIXING COMPLETE")
        print("="*50)
        
        print("📈 Successfully fixed:")
        for symbol, rows in results.items():
            print(f"   {symbol}: {rows:,} M5 candles")
    
    else:
        print("❌ No valid data found")

if __name__ == "__main__":
    main()