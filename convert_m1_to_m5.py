#!/usr/bin/env python3
"""
Convert M1 (1-minute) data to M5 (5-minute) data
Combines all available M1 data and converts to M5 timeframe
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_m1_to_m5(df):
    """
    Convert M1 data to M5 data using OHLCV aggregation
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Resample to 5-minute intervals
    m5_data = df.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to get timestamp as column
    m5_data.reset_index(inplace=True)
    
    return m5_data

def process_symbol_data(symbol):
    """
    Process all M1 data files for a specific symbol and convert to M5
    """
    logger.info(f"Processing {symbol} data...")
    
    # Find all M1 converted files for this symbol
    pattern = f"train_data/DAT_ASCII_{symbol}_M1_*_converted.csv"
    files = glob.glob(pattern)
    
    if not files:
        logger.warning(f"No M1 data files found for {symbol}")
        return None
    
    logger.info(f"Found {len(files)} files for {symbol}")
    
    # Read and combine all files
    all_data = []
    for file in sorted(files):
        try:
            logger.info(f"Reading {file}")
            df = pd.read_csv(file)
            
            # Basic validation
            if len(df) == 0:
                logger.warning(f"Empty file: {file}")
                continue
                
            # Check required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns in {file}: {df.columns.tolist()}")
                continue
            
            all_data.append(df)
            logger.info(f"Loaded {len(df)} rows from {file}")
            
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue
    
    if not all_data:
        logger.error(f"No valid data found for {symbol}")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} rows")
    
    # Remove duplicates and sort
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    logger.info(f"After deduplication: {len(combined_df)} rows")
    
    # Convert to M5
    m5_data = convert_m1_to_m5(combined_df.copy())
    logger.info(f"M5 data: {len(m5_data)} rows")
    
    # Save M5 data
    output_file = f"train_data/{symbol}_M5_combined.csv"
    m5_data.to_csv(output_file, index=False)
    logger.info(f"Saved M5 data to {output_file}")
    
    # Print data summary
    print(f"\n📊 {symbol} Data Summary:")
    print(f"   Original M1 rows: {len(combined_df):,}")
    print(f"   Converted M5 rows: {len(m5_data):,}")
    print(f"   Date range: {m5_data['timestamp'].min()} to {m5_data['timestamp'].max()}")
    print(f"   Output file: {output_file}")
    
    return m5_data

def main():
    """
    Main function to convert all available M1 data to M5
    """
    print("🔄 Converting M1 data to M5 timeframe...")
    print("="*60)
    
    # Find all available symbols
    pattern = "train_data/DAT_ASCII_*_M1_*_converted.csv"
    files = glob.glob(pattern)
    
    # Extract unique symbols
    symbols = set()
    for file in files:
        # Extract symbol from filename: DAT_ASCII_XAUUSD_M1_2024_converted.csv
        parts = os.path.basename(file).split('_')
        if len(parts) >= 3:
            symbol = parts[2]  # XAUUSD
            symbols.add(symbol)
    
    if not symbols:
        print("❌ No M1 data files found!")
        return
    
    print(f"Found symbols: {', '.join(sorted(symbols))}")
    print()
    
    # Process each symbol
    results = {}
    for symbol in sorted(symbols):
        try:
            m5_data = process_symbol_data(symbol)
            if m5_data is not None:
                results[symbol] = len(m5_data)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE")
    print("="*60)
    
    if results:
        print("📈 Successfully converted:")
        for symbol, rows in results.items():
            print(f"   {symbol}: {rows:,} M5 candles")
        
        print(f"\n📁 Output files saved in train_data/ folder")
        print(f"   Format: {list(results.keys())[0]}_M5_combined.csv")
        
        # Create a combined summary file
        summary_data = []
        for symbol in results.keys():
            file_path = f"train_data/{symbol}_M5_combined.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                summary_data.append({
                    'symbol': symbol,
                    'rows': len(df),
                    'start_date': df['timestamp'].min(),
                    'end_date': df['timestamp'].max(),
                    'file': file_path
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('train_data/M5_data_summary.csv', index=False)
            print(f"📋 Summary saved to: train_data/M5_data_summary.csv")
    else:
        print("❌ No data was successfully converted")

if __name__ == "__main__":
    main()