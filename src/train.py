import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
import joblib
from .models import BaseModel, StackedModel, VolatilityModel
from .utils import save_json

logger = logging.getLogger(__name__)

def walk_forward_split(df: pd.DataFrame, train_window: int, val_window: int, step: int = 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward splits."""
    splits = []
    n = len(df)
    for i in range(train_window, n - val_window, step):
        train_end = i
        val_end = i + val_window
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        splits.append((train, val))
    return splits

def train_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Train on one fold."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    # Simple Ridge for now
    model = BaseModel('ridge')
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return preds, y_val.values

def train_walk_forward(df: pd.DataFrame, target_col: str, feature_cols: List[str], train_window: int, val_window: int, n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Train with walk-forward CV."""
    splits = walk_forward_split(df, train_window, val_window)
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(train_fold)(train, val, target_col, feature_cols) for train, val in splits
    )
    all_preds = np.concatenate([r[0] for r in results])
    all_true = np.concatenate([r[1] for r in results])
    return all_preds, all_true

def train_volatility_model(df: pd.DataFrame, vol_target: str, feature_cols: List[str], config: Dict[str, Any] = None) -> VolatilityModel:
    """Train volatility model."""
    if config is None:
        config = {}
    model_config = config.get('model', {})
    model = VolatilityModel(max_depth=model_config.get('max_depth', 6), n_estimators=model_config.get('n_estimators', 100))
    model.fit(df[feature_cols], df[vol_target])
    return model

def train_stacked_model(df: pd.DataFrame, target_col: str, feature_cols: List[str], config: Dict[str, Any] = None) -> StackedModel:
    """Train stacked model."""
    if config is None:
        config = {}
    model_config = config.get('model', {})
    
    base_models = [
        BaseModel('ridge'),
        BaseModel('nystroem_ridge'),
        BaseModel('lightgbm', max_depth=model_config.get('max_depth', 6), n_estimators=model_config.get('n_estimators', 100))
    ]
    meta_model = BaseModel('lightgbm', max_depth=model_config.get('max_depth', 6), n_estimators=model_config.get('n_estimators', 100))
    stacked = StackedModel(base_models, meta_model)
    stacked.fit(df[feature_cols], df[target_col])
    return stacked