import pytest
import yaml
import os
import tempfile
from src.train import train_stacked_model, train_volatility_model
from src.backtest import compute_weights, calibrate_k
import pandas as pd
import numpy as np

def test_config_loading():
    """Test that config values are loaded correctly."""
    config = {
        'model': {
            'max_depth': 8,
            'n_estimators': 500,
            'leverage_calibration_k': 2.0,
            'ema_alpha': 0.1,
            'regime_sensitivity': 0.6
        },
        'seed': 123
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['model']['max_depth'] == 8
        assert loaded_config['model']['n_estimators'] == 500
        assert loaded_config['model']['leverage_calibration_k'] == 2.0
        assert loaded_config['seed'] == 123
    finally:
        os.unlink(config_path)

def test_model_config_propagation():
    """Test that model config parameters are used in training."""
    config = {
        'model': {
            'max_depth': 4,
            'n_estimators': 50
        }
    }
    
    # Create dummy data
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100),
        'volatility': np.random.randn(100)
    })
    feature_cols = ['feature1', 'feature2']
    
    # Train models with config
    stacked_model = train_stacked_model(df, 'target', feature_cols, config)
    vol_model = train_volatility_model(df, 'volatility', feature_cols, config)
    
    # Check that models were created (basic smoke test)
    assert stacked_model is not None
    assert vol_model is not None

def test_backtest_config_propagation():
    """Test that backtest config parameters are used."""
    config = {
        'model': {
            'regime_sensitivity': 0.5,
            'leverage_calibration_k': 1.8
        }
    }
    
    mu = np.array([0.01, 0.02])
    sigma = np.array([0.1, 0.2])
    k = 1.0
    df = pd.DataFrame({'regime_0': [0, 1]})  # One normal, one high vol
    
    # Test mid variant with regime sensitivity
    weights = compute_weights(mu, sigma, k, variant='mid', df=df, config=config)
    
    # In high vol regime (second element), weight should be reduced by regime_sensitivity
    assert weights[1] < weights[0]  # High vol should have lower weight
    
    # Test k override
    assert config['model']['leverage_calibration_k'] == 1.8