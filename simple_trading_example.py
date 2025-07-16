#!/usr/bin/env python3
"""
Simple FinRL Trading Example
This example demonstrates how to:
1. Download stock data
2. Train a reinforcement learning agent
3. Test the agent's performance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import FinRL components
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

def download_data():
    """Download stock data using Yahoo Finance"""
    print("📊 Downloading stock data...")
    
    # Define date ranges
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    # Simple stock list (you can modify this)
    ticker_list = ["AAPL", "MSFT", "GOOGL"]
    
    # Download data using DataProcessor
    dp = DataProcessor(data_source="yahoofinance")
    data = dp.download_data(ticker_list, start_date, end_date, time_interval="1D")
    data = dp.clean_data(data)
    
    # Add technical indicators (simplified list to avoid TA-lib dependency)
    tech_indicators = ["rsi_30", "close_30_sma", "close_60_sma"]
    data = dp.add_technical_indicator(data, tech_indicators)
    
    print(f"✅ Downloaded data for {len(ticker_list)} stocks")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📈 Data shape: {data.shape}")
    
    return data, ticker_list

def prepare_environment(data):
    """Prepare the trading environment"""
    print("🏗️ Preparing trading environment...")
    
    # Convert data to arrays for the environment
    dp = DataProcessor(data_source="yahoofinance")
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix=False)
    
    # Create environment configuration
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    
    # Create environment instance
    env_instance = StockTradingEnv(config=env_config)
    
    print("✅ Environment prepared successfully")
    return env_instance

def train_agent(env_instance):
    """Train the reinforcement learning agent"""
    print("🤖 Training RL agent...")
    
    # Create DRL agent
    agent = DRLAgent(env=env_instance)
    
    # Get PPO model (Proximal Policy Optimization)
    model = agent.get_model("ppo")
    
    # Train the model (reduced timesteps for quick demo)
    print("🏋️ Starting training... (this may take a few minutes)")
    trained_model = agent.train_model(
        model=model, 
        tb_log_name="ppo_trading", 
        total_timesteps=10000  # Reduced for demo
    )
    
    # Save the trained model
    trained_model.save("./trained_ppo_model")
    print("✅ Training completed and model saved!")
    
    return trained_model

def test_agent(data, ticker_list):
    """Test the trained agent"""
    print("🧪 Testing trained agent...")
    
    # Split data for testing (last 6 months)
    test_data = data[data.date >= '2023-07-01'].reset_index(drop=True)
    
    # Prepare test environment
    dp = DataProcessor(data_source="yahoofinance")
    price_array, tech_array, turbulence_array = dp.df_to_array(test_data, if_vix=False)
    
    test_env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    
    test_env = StockTradingEnv(config=test_env_config)
    
    # Load trained model and test
    from stable_baselines3 import PPO
    model = PPO.load("./trained_ppo_model")
    
    # Run test
    obs = test_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
    
    # Calculate performance
    final_portfolio_value = test_env.asset_memory[-1]
    initial_portfolio_value = test_env.asset_memory[0]
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
    
    print("📊 Test Results:")
    print(f"💰 Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
    print(f"💰 Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"📈 Total Return: {total_return:.2f}%")
    print(f"🎯 Total Reward: {total_reward:.2f}")
    
    return total_return

def main():
    """Main function to run the complete example"""
    print("🚀 Welcome to FinRL - Your AI Trading Assistant!")
    print("=" * 50)
    
    try:
        # Step 1: Download data
        data, ticker_list = download_data()
        
        # Step 2: Prepare environment
        env_instance = prepare_environment(data)
        
        # Step 3: Train agent
        trained_model = train_agent(env_instance)
        
        # Step 4: Test agent
        total_return = test_agent(data, ticker_list)
        
        print("=" * 50)
        print("🎉 FinRL Example Completed Successfully!")
        print(f"🏆 Your AI agent achieved {total_return:.2f}% return!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("💡 This might be due to missing dependencies or data issues.")

if __name__ == "__main__":
    main()