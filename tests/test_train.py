import pytest
import pandas as pd
import numpy as np
from src.train import walk_forward_split

def test_walk_forward_split():
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({'value': np.random.randn(100)}, index=dates)
    splits = walk_forward_split(df, 50, 10, 10)
    assert len(splits) > 0
    for train, val in splits:
        assert len(train) == 50
        assert len(val) == 10
        assert train.index[-1] < val.index[0]