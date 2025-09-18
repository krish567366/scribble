import pytest
import pandas as pd
import numpy as np
from src.features import compute_rolling_returns, compute_volatility

def test_compute_rolling_returns():
    df = pd.DataFrame({'price': [100, 101, 102, 103, 104]})
    returns = compute_rolling_returns(df, [1, 2])
    assert 'return_1d' in returns.columns
    assert 'return_2d' in returns.columns
    assert len(returns) == len(df)
    assert not returns.isnull().any().any()

def test_compute_volatility():
    df = pd.DataFrame({'price': [100, 101, 102, 103, 104]})
    vol = compute_volatility(df, 2)
    assert len(vol) == len(df)
    assert vol.iloc[0] == 0  # First value NaN filled or 0