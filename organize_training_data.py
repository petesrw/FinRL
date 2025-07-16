#!/usr/bin/env python3
"""
Organize and standardize training data by symbol
Convert all M5 data to consistent format and organize by folders
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

def standardize_columns(df, symbol):
    """
    Standardize column names and format
    """
    # Map different column name variations to standard names
    column_mapping = {
        'Time': 'timestamp',
        'time': 'timestamp',
        'timestamp': 'timestamp',
        'Date': 'timestamp',
        'datetime': 'timestamp',
        
        'Open': 'open',
        'open': 'open',
        'OPEN': 'open',
        
        'High': 'high',
        'high': 'high',
        'HIGH': 'high',
        
        'Low': 'low',
        'low': 'low',
        'LOW': 'low',
        
        'Close': 'close',
        'close': 'close',
        'CLOSE': 'close',
        
        'Volume': 'volume',
        'volume': 'volume',
        'VOLUME': 'volume',
        'Vol': 'volume'
    }
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    
    # Ensure we have required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_renamed.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns for {symbol}: {missing_cols}")
        return None
    
    # Add volume if missing
    if 'volume' not in df_renamed.columns:
        df_renamed['volume'] = 0
        logger.info(f"Added default volume column for {symbol}")
    
    # Select and reorder columns
    standard_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_standard = df_renamed[standard_cols].copy()
    
    # Convert timestamp to datetime
    df_standard['timestamp'] = pd.to_datetime(df_standard['timestamp'])
    
    # Convert price columns to float
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df_standard[col] = pd.to_numeric(df_standard[col], errors='coerce')
    
    # Convert volume to int
    df_standard['volume'] = pd.to_numeric(df_standard['volume'], errors='coerce').fillna(0).astype(int)
    
    # Remove rows with NaN prices
    df_standard = df_standard.dropna(subset=price_cols)
    
    # Sort by timestamp
    df_standard = df_standard.sort_values('timestamp').reset_index(drop=True)
    
    return df_standard

def process_symbol_folder(symbol):
    """
    Process M5 data for a specific symbol
    """
    logger.info(f"Processing {symbol}...")
    
    # Look for M5 data file
    m5_file = f"train_data/{symbol}/{symbol}_M5.csv"
    
    if not os.path.exists(m5_file):
        logger.warning(f"M5 file not found: {m5_file}")
        return None
    
    try:
        # Read the data
        logger.info(f"Reading {m5_file}")
        
        # Try different separators
        separators = ['\t', ',', ';']
        df = None
        
        for sep in separators:
            try:
                df = pd.read_csv(m5_file, sep=sep)
                if len(df.columns) >= 5:  # Should have at least OHLC + timestamp
                    logger.info(f"Successfully read with separator '{sep}'")
                    # Remove extra columns if they exist (some files have extra columns)
                    expected_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if len(df.columns) > 6:
                        df = df.iloc[:, :6]  # Keep only first 6 columns
                        df.columns = expected_cols
                    break
            except:
                continue
        
        if df is None or len(df) == 0:
            logger.error(f"Could not read data from {m5_file}")
            return None
        
        logger.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Standardize the data
        df_standard = standardize_columns(df, symbol)
        
        if df_standard is None:
            return None
        
        logger.info(f"Standardized data: {len(df_standard)} rows")
        
        # Validate data quality
        if len(df_standard) < 100:
            logger.warning(f"Very little data for {symbol}: {len(df_standard)} rows")
        
        # Check for reasonable price ranges
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if df_standard[col].min() <= 0:
                logger.warning(f"Found non-positive prices in {symbol} {col}")
        
        # Save standardized data
        output_file = f"train_data/{symbol}/{symbol}_M5_standardized.csv"
        df_standard.to_csv(output_file, index=False)
        logger.info(f"Saved standardized data to {output_file}")
        
        # Print summary
        print(f"\n📊 {symbol} Data Summary:")
        print(f"   Rows: {len(df_standard):,}")
        print(f"   Date range: {df_standard['timestamp'].min()} to {df_standard['timestamp'].max()}")
        print(f"   Price range: {df_standard['close'].min():.5f} - {df_standard['close'].max():.5f}")
        print(f"   Output: {output_file}")
        
        return df_standard
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def create_combined_dataset():
    """
    Create a combined dataset with all symbols
    """
    logger.info("Creating combined dataset...")
    
    # Find all symbol folders
    symbol_folders = [d for d in os.listdir('train_data') if os.path.isdir(f'train_data/{d}') and d.isupper()]
    
    combined_data = []
    
    for symbol in sorted(symbol_folders):
        standardized_file = f"train_data/{symbol}/{symbol}_M5_standardized.csv"
        
        if os.path.exists(standardized_file):
            try:
                df = pd.read_csv(standardized_file)
                df['symbol'] = symbol
                combined_data.append(df)
                logger.info(f"Added {symbol}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error reading {standardized_file}: {e}")
    
    if combined_data:
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Sort by symbol and timestamp
        combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Save combined dataset
        output_file = "train_data/all_symbols_M5_combined.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n📈 Combined Dataset:")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Symbols: {', '.join(sorted(combined_df['symbol'].unique()))}")
        print(f"   Output: {output_file}")
        
        # Create summary by symbol
        summary_data = []
        for symbol in sorted(combined_df['symbol'].unique()):
            symbol_data = combined_df[combined_df['symbol'] == symbol]
            summary_data.append({
                'symbol': symbol,
                'rows': len(symbol_data),
                'start_date': symbol_data['timestamp'].min(),
                'end_date': symbol_data['timestamp'].max(),
                'min_price': symbol_data['close'].min(),
                'max_price': symbol_data['close'].max(),
                'avg_price': symbol_data['close'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('train_data/training_data_summary.csv', index=False)
        
        return combined_df
    
    return None

def main():
    """
    Main function to organize all training data
    """
    print("🔄 Organizing and standardizing training data...")
    print("="*60)
    
    # Find all symbol folders
    symbol_folders = [d for d in os.listdir('train_data') if os.path.isdir(f'train_data/{d}') and d.isupper()]
    
    if not symbol_folders:
        print("❌ No symbol folders found!")
        return
    
    print(f"Found symbol folders: {', '.join(sorted(symbol_folders))}")
    print()
    
    # Process each symbol
    results = {}
    for symbol in sorted(symbol_folders):
        try:
            df = process_symbol_folder(symbol)
            if df is not None:
                results[symbol] = len(df)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Create combined dataset
    if results:
        print("\n" + "="*40)
        print("Creating combined dataset...")
        print("="*40)
        
        combined_df = create_combined_dataset()
        
        if combined_df is not None:
            print("\n" + "="*60)
            print("✅ DATA ORGANIZATION COMPLETE")
            print("="*60)
            
            print("📈 Successfully processed:")
            for symbol, rows in results.items():
                print(f"   {symbol}: {rows:,} M5 candles")
            
            print(f"\n📁 Files created:")
            print(f"   Individual: train_data/[SYMBOL]/[SYMBOL]_M5_standardized.csv")
            print(f"   Combined: train_data/all_symbols_M5_combined.csv")
            print(f"   Summary: train_data/training_data_summary.csv")
            
        else:
            print("❌ Failed to create combined dataset")
    else:
        print("❌ No data was successfully processed")

if __name__ == "__main__":
    main()