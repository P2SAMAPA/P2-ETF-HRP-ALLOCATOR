"""
Main training script for HRP Allocator engine.
Computes HRP weights for all universes and pushes results to Hugging Face.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from hrp_model import HRPAllocator
import push_results

def run_hrp_allocation():
    """Orchestrates the full HRP allocation pipeline."""
    
    print(f"=== P2-ETF-HRP-ALLOCATOR Run: {config.TODAY} ===")
    
    # Load master data
    df_master = data_manager.load_master_data()
    
    # Initialize allocator
    allocator = HRPAllocator(linkage_method=config.LINKAGE_METHOD)
    
    all_weights = {}
    cluster_info = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        
        # Prepare returns matrix
        returns_matrix = data_manager.prepare_returns_matrix(df_master, tickers)
        
        if len(returns_matrix) < config.MIN_OBSERVATIONS:
            print(f"    Insufficient data: {len(returns_matrix)} observations")
            continue
        
        # Use recent window
        recent_returns = returns_matrix.iloc[-config.LOOKBACK_WINDOW:]
        print(f"    Using {len(recent_returns)} observations")
        
        # Compute HRP weights
        weights = allocator.allocate(recent_returns)
        all_weights[universe_name] = weights
        
        # Print summary
        top_3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top 3 allocations: {', '.join([f'{t}: {w:.2%}' for t, w in top_3])}")
        
        # Store cluster linkage and original tickers for visualization
        linkage, original_tickers = allocator.get_linkage_and_labels()
        if linkage is not None and original_tickers is not None:
            cluster_info[universe_name] = {
                "linkage": linkage.tolist(),
                "original_tickers": original_tickers
            }
    
    # Build output payload
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "linkage_method": config.LINKAGE_METHOD
        },
        "weights": all_weights,
        "cluster_info": cluster_info
    }
    
    # Push to Hugging Face
    push_results.push_daily_result(output_payload)
    
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_hrp_allocation()
