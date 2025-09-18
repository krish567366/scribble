#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml
from src.utils import setup_logging, set_seeds, save_json, get_git_commit, timestamp
from src.data import get_cached_data
from src.features import create_features, save_features, load_features
from src.train import train_stacked_model, train_volatility_model
from src.backtest import compute_weights, calibrate_k, simulate_portfolio, calculate_sharpe, stress_test, plot_performance, apply_ema_smoothing
from src.models import save_model
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run forecasting pipeline')
    parser.add_argument('--config', default='configs/baseline.yaml', help='Path to config YAML file')
    parser.add_argument('--data-root', default='/data', help='Data root directory')
    parser.add_argument('--rebuild-cache', action='store_true', help='Rebuild cache')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for LightGBM')
    parser.add_argument('--variant', choices=['conservative', 'aggressive', 'mid'], default='conservative', help='Model variant')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--train-window', type=int, default=2520, help='Training window size')
    parser.add_argument('--val-window', type=int, default=252, help='Validation window size')
    parser.add_argument('--k', type=float, default=None, help='Leverage calibration factor')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging()
    set_seeds(config.get('seed', 42))
    
    # Paths
    cache_path = 'cache/processed_data.parquet'
    features_path = 'cache/features.parquet'
    artifacts_dir = 'artifacts/'
    submissions_dir = 'submissions/'
    
    # Ensure directories exist
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    # Load data
    df = get_cached_data(cache_path, args.rebuild_cache, args.data_root)
    
    # Create features
    if args.rebuild_cache or not os.path.exists(features_path):
        df_feat, norm_params = create_features(df)
        save_features(df_feat, features_path)
        save_json(norm_params, os.path.join(artifacts_dir, 'norm_params.json'))
    else:
        df_feat = load_features(features_path)
    
    # Assume target columns
    target_col = 'market_forward_excess_returns'
    vol_target = 'volatility'
    feature_cols = [c for c in df_feat.columns if c not in [target_col, vol_target]]
    
    # Train models
    stacked_model = train_stacked_model(df_feat, target_col, feature_cols, config)
    vol_model = train_volatility_model(df_feat, vol_target, feature_cols, config)
    
    # Save models
    save_model(stacked_model, os.path.join(artifacts_dir, 'stacked_model.pkl'))
    save_model(vol_model, os.path.join(artifacts_dir, 'vol_model.pkl'))
    
    # Predict
    mu = stacked_model.predict(df_feat[feature_cols])
    sigma = vol_model.predict(df_feat[feature_cols])
    
    # Calibrate k if not provided
    if args.k is None:
        returns = df_feat[target_col]  # Assume this is returns
        args.k = calibrate_k(returns, mu, sigma, variant=args.variant, df=df_feat, config=config)
    
    # Override k from config if specified
    if 'leverage_calibration_k' in config['model']:
        args.k = config['model']['leverage_calibration_k']
    
    # Compute weights
    weights = compute_weights(mu, sigma, args.k, args.variant, df_feat, config)
    
    # Apply EMA smoothing
    ema_alpha = config['model'].get('ema_alpha', 0.1)
    weights = apply_ema_smoothing(weights, alpha=ema_alpha)
    
    # Simulate and metrics
    port_returns = simulate_portfolio(df_feat[target_col], weights)
    sharpe = calculate_sharpe(port_returns)
    
    # Stress tests
    stress_periods = [('2008_crisis', '2008-01-01', '2009-12-31'), ('2020_crash', '2020-01-01', '2021-12-31')]
    stress_results = stress_test(df_feat[target_col], weights, stress_periods)
    
    # Plot
    plot_path = os.path.join(artifacts_dir, 'performance.png')
    plot_performance(df_feat[target_col], weights, plot_path)
    
    # Save submission
    submission = pd.DataFrame({'date': df_feat.index, 'weight': weights})
    submission_path = os.path.join(submissions_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    # Save report
    report = {
        'config': args.config,
        'variant': args.variant,
        'k': args.k,
        'sharpe': sharpe,
        'stress_tests': stress_results,
        'git_commit': get_git_commit(),
        'timestamp': timestamp()
    }
    save_json(report, os.path.join(artifacts_dir, 'run_report.json'))
    
    logger.info(f"Pipeline completed. Submission saved to {submission_path}")

if __name__ == '__main__':
    main()