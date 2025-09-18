import pytest
import numpy as np
from src.backtest import compute_weights

def test_compute_weights():
    mu = np.array([0.01, 0.02, -0.01])
    sigma = np.array([0.1, 0.2, 0.05])
    k = 1.0
    weights = compute_weights(mu, sigma, k)
    assert len(weights) == len(mu)
    assert np.all(weights >= 0)
    assert np.all(weights <= 2)
    # Test clipping
    assert weights[2] == 0  # Negative mu should be 0