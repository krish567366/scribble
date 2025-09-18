#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml
from src.utils import setup_logging, set_seeds, save_json, get_git_commit, timestamp
from src.data import get_cached_data
from src.features import create_features, save_features, load_features
from src.backtest import compute_weights, calibrate_k, simulate_portfolio, calculate_sharpe, stress_test, plot_performance, apply_ema_smoothing
from src.models import save_model, TechnicalStrategy
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run forecasting pipeline')
    parser.add_argument('--config', default='configs/baseline.yaml', help='Path to config YAML file')
    parser.add_argument('--data-root', default='data', help='Data root directory')
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
    
    # Select features: exclude target columns and columns with too many NaNs or no variance
    excluded_cols = [target_col, vol_target, 'forward_returns', 'risk_free_rate', 
                     'lagged_forward_returns', 'lagged_risk_free_rate', 'lagged_market_forward_excess_returns',
                     'return_1d', 'return_5d', 'return_21d']  # Exclude return features to avoid data leakage
    potential_features = [c for c in df_feat.columns if c not in excluded_cols]
    feature_cols = []
    for col in potential_features:
        if df_feat[col].isna().sum() / len(df_feat) < 0.5:  # Less than 50% NaN
            if df_feat[col].nunique() > 1:  # Has variance (not constant)
                feature_cols.append(col)
    
    logger.info(f"Selected {len(feature_cols)} features out of {len(potential_features)} potential features")
    
    # Save selected features list for consistent prediction
    save_json({'selected_features': feature_cols}, os.path.join(artifacts_dir, 'selected_features.json'))
    
    # Use technical strategy instead of ML models
    strategy = TechnicalStrategy(config)
    
    # Train ML correction component
    if hasattr(df_feat, target_col) and target_col in df_feat.columns:
        strategy.fit_ml_correction(df_feat[feature_cols], df_feat[target_col])
    
    # Save strategy (just save the config for reproducibility)
    save_model(strategy, os.path.join(artifacts_dir, 'technical_strategy.pkl'))
    
    # Predict using technical indicators
    mu = strategy.predict_mu(df_feat[feature_cols])
    sigma = strategy.predict_sigma(df_feat[feature_cols])

    # Ensure mu and sigma are 1D arrays
    if hasattr(mu, 'flatten'):
        mu = mu.flatten()
    if hasattr(sigma, 'flatten'):
        sigma = sigma.flatten()

    # Ensure they are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    logger.info(f"Predictions shape - mu: {mu.shape}, sigma: {sigma.shape}")
    
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
    if ema_alpha > 0:
        weights = apply_ema_smoothing(weights, alpha=ema_alpha)
    
    # Simulate and metrics
    port_returns = simulate_portfolio(df[target_col], weights)
    sharpe = calculate_sharpe(port_returns)
    
    # Stress tests - comprehensive period analysis
    # Define periods based on data length (assuming ~9000 samples)
    total_samples = len(df)
    early_end = int(total_samples * 0.3)    # First 30%
    mid_start = int(total_samples * 0.3)
    mid_end = int(total_samples * 0.7)      # 30-70%
    late_start = int(total_samples * 0.7)   # Last 30%
    
    stress_periods = [
        ('early_period', df.index[0], df.index[early_end-1]),
        ('mid_period', df.index[mid_start], df.index[mid_end-1]), 
        ('late_period', df.index[late_start], df.index[-1])
    ]
    
    stress_results = stress_test(df[target_col], weights, stress_periods)
    
    # Enhanced validation against targets
    validation_results = {
        'overall_sharpe': sharpe,
        'early_sharpe': stress_results['early_period']['sharpe'],
        'mid_sharpe': stress_results['mid_period']['sharpe'],
        'late_sharpe': stress_results['late_period']['sharpe'],
        'max_drawdown': min([result['max_drawdown'] for result in stress_results.values()]),
        'targets': {
            'overall_sharpe_target': 0.36,
            'late_sharpe_target': 0.1,
            'max_drawdown_target': -0.25
        }
    }
    
    # Log validation results
    logger.info(f"Validation Results:")
    logger.info(f"Overall Sharpe: {sharpe:.3f} (Target: {validation_results['targets']['overall_sharpe_target']:.3f})")
    logger.info(f"Early Sharpe: {stress_results['early_period']['sharpe']:.3f}")
    logger.info(f"Mid Sharpe: {stress_results['mid_period']['sharpe']:.3f}")
    logger.info(f"Late Sharpe: {stress_results['late_period']['sharpe']:.3f} (Target: {validation_results['targets']['late_sharpe_target']:.3f})")
    logger.info(f"Max Drawdown: {validation_results['max_drawdown']:.3f} (Target: {validation_results['targets']['max_drawdown_target']:.3f})")
    
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
        'validation': validation_results,
        'git_commit': get_git_commit(),
        'timestamp': timestamp()
    }
    save_json(report, os.path.join(artifacts_dir, 'run_report.json'))
    
    logger.info(f"Pipeline completed. Submission saved to {submission_path}")

if __name__ == '__main__':
    main()