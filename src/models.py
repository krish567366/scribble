import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from typing import List, Dict, Any
import joblib

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.model = self._build_model(**kwargs)
    
    def _build_model(self, **kwargs):
        if self.model_type == 'ridge':
            return Ridge(**kwargs)
        elif self.model_type == 'nystroem_ridge':
            return Pipeline([
                ('nystroem', Nystroem(**kwargs.get('nystroem', {}))),
                ('ridge', Ridge(**kwargs.get('ridge', {})))
            ])
        elif self.model_type == 'lightgbm':
            return LGBMRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class StackedModel:
    def __init__(self, base_models: List[BaseModel], meta_model: BaseModel):
        self.base_models = base_models
        self.meta_model = meta_model
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Train base models and get OOF predictions
        oof_preds = []
        for model in self.base_models:
            model.fit(X, y)
            oof_preds.append(model.predict(X))
        oof_df = pd.DataFrame(np.column_stack(oof_preds), index=X.index)
        
        # Train meta model
        self.meta_model.fit(oof_df, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = [model.predict(X) for model in self.base_models]
        base_df = pd.DataFrame(np.column_stack(base_preds), index=X.index)
        return self.meta_model.predict(base_df)

class VolatilityModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__('lightgbm', **kwargs)  # Use LightGBM for volatility

def save_model(model, path: str):
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path: str):
    return joblib.load(path)