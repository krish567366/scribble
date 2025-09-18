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

class TechnicalStrategy:
    """Rule-based strategy with weighted combination of normalized features."""
    
    def __init__(self, config=None):
        self.config = config or {}
        # Define feature weights for the rule-based backbone
        self.feature_weights = {
            'M1': 0.25,      # Macro feature 1
            'M13': 0.20,    # Macro feature 13
            'P2': 0.20,     # Price feature 2
            'E3': 0.10,     # Economic feature 3
            'E2': 0.10,     # Economic feature 2
            'I2': 0.08,     # Indicator feature 2
            'S12': 0.07     # Sector feature 12
        }
        
        # Initialize ML correction model
        self.ml_correction = ML_Correction(alpha=1.0)
        self.correction_weight = 0.1  # Small contribution from ML
        
        # Initialize mid-period ensemble model
        self.mid_period_ensemble = MidPeriodEnsemble(alpha=0.1)
        self.mid_period_weight = 0.15  # Weight for mid-period ensemble
        
    def predict_mu(self, features_df):
        """Predict expected returns using rule-based weighted combination."""
        signal = np.zeros(len(features_df))
        
        # Rule-based backbone: weighted combination of normalized features
        total_weight = 0
        for feature, weight in self.feature_weights.items():
            if feature in features_df.columns:
                # Use z-score normalized features if available, otherwise raw normalized
                zscore_col = f'{feature}_zscore'
                if zscore_col in features_df.columns:
                    feat_values = features_df[zscore_col].fillna(0)
                else:
                    # Fallback to raw feature normalization
                    feat_values = features_df[feature].fillna(0)
                    if feat_values.std() > 0:
                        feat_values = (feat_values - feat_values.mean()) / feat_values.std()
                
                signal += weight * feat_values.values
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            signal /= total_weight
        
        # Add trend component
        trend_signal = self._compute_trend_signal(features_df)
        signal += 0.3 * trend_signal  # 30% weight on trend
        
        # Add interaction features
        interaction_signal = self._compute_interaction_signal(features_df)
        signal += 0.1 * interaction_signal  # 10% weight on interactions

        # Add mid-period specific features
        mid_period_signal = self._compute_mid_period_signal(features_df)
        signal += 0.2 * mid_period_signal  # 20% weight on mid-period features
        
        # Add mid-period ensemble signal
        ensemble_signal = self._get_mid_period_ensemble_signal(features_df)
        signal += 0.15 * ensemble_signal  # 15% weight on ensemble

        # Apply drawdown scaling
        drawdown_scaling = self._compute_drawdown_scaling(features_df)
        signal *= drawdown_scaling

        # Normalize to [-1, 1] range
        signal = np.clip(signal, -1, 1)
        
        # Apply ML correction if trained
        ml_correction = self._get_ml_correction(features_df)
        signal += self.correction_weight * ml_correction
        
        # Ensure signal is a numpy array
        signal = np.asarray(signal).flatten()
        
        return signal
    
    def fit_ml_correction(self, features_df, target_returns):
        """Train the ML correction model."""
        # Use a subset of features for ML correction
        ml_features = []
        self.ml_feature_cols = []  # Store which features were used for training
        
        feature_cols = ['M1', 'M13', 'P2', 'E3', 'E2', 'I2', 'S12', 
                       'roc_5d', 'roc_21d', 'volatility']
        
        for col in feature_cols:
            if col in features_df.columns:
                ml_features.append(features_df[col].fillna(0).values.reshape(-1, 1))
                self.ml_feature_cols.append(col)
        
        if ml_features:
            X = np.column_stack(ml_features)
            # Target is the residual between actual returns and rule-based prediction
            rule_pred = self.predict_mu(features_df)
            residual = target_returns - rule_pred
            
            # Remove NaN values for training
            valid_mask = ~(np.isnan(residual) | np.isnan(X).any(axis=1))
            if valid_mask.sum() > 100:  # Need minimum samples
                X_clean = X[valid_mask]
                y_clean = residual[valid_mask]
                self.ml_correction.fit(X_clean, y_clean)
        
        # Fit mid-period ensemble
        self.fit_mid_period_ensemble(features_df, target_returns)
    
    def _get_ml_correction(self, features_df):
        """Get ML correction prediction."""
        if not hasattr(self, 'ml_feature_cols') or not self.ml_correction.is_trained:
            return np.zeros(len(features_df))
            
        ml_features = []
        
        for col in self.ml_feature_cols:
            if col in features_df.columns:
                ml_features.append(features_df[col].fillna(0).values.reshape(-1, 1))
        
        if ml_features and len(ml_features) == len(self.ml_feature_cols):
            X = np.column_stack(ml_features)
            correction = self.ml_correction.predict(X)
            return np.asarray(correction).flatten()
        else:
            return np.zeros(len(features_df))
    
    def _compute_trend_signal(self, features_df):
        """Compute trend-based signal component."""
        trend_signal = np.zeros(len(features_df))
        
        # 5-day momentum (short-term)
        if 'roc_5d' in features_df.columns:
            roc_5d = features_df['roc_5d'].fillna(0)
            trend_signal += 0.6 * np.sign(roc_5d.values)
        
        # 21-day momentum (medium-term)
        if 'roc_21d' in features_df.columns:
            roc_21d = features_df['roc_21d'].fillna(0)
            trend_signal += 0.4 * np.sign(roc_21d.values)
        
        return trend_signal
    
    def _compute_interaction_signal(self, features_df):
        """Compute interaction-based signal component."""
        interaction_signal = np.zeros(len(features_df))
        
        # M1/M13 ratio
        if 'M1_M13_ratio' in features_df.columns:
            ratio = features_df['M1_M13_ratio'].fillna(1)
            # Mean-reversion on extreme ratios
            ratio_zscore = (ratio - ratio.rolling(63).mean()) / (ratio.rolling(63).std() + 1e-8)
            ratio_zscore = ratio_zscore.fillna(0).values  # Convert to numpy array
            interaction_signal += 0.5 * np.clip(-ratio_zscore, -1, 1)
        
        # P2/E2 ratio
        if 'P2_E2_ratio' in features_df.columns:
            ratio = features_df['P2_E2_ratio'].fillna(1)
            ratio_zscore = (ratio - ratio.rolling(63).mean()) / (ratio.rolling(63).std() + 1e-8)
            ratio_zscore = ratio_zscore.fillna(0).values  # Convert to numpy array
            interaction_signal += 0.5 * np.clip(-ratio_zscore, -1, 1)
        
        return interaction_signal
    
    def _compute_mid_period_signal(self, features_df):
        """Compute mid-period specific signal components."""
        mid_signal = np.zeros(len(features_df))
        
        # Medium-term momentum (42-day)
        if 'roc_42d' in features_df.columns:
            roc_42d = features_df['roc_42d'].fillna(0).values
            mid_signal += 0.25 * np.sign(roc_42d)
        
        # Volatility-adjusted momentum
        if 'momentum_vol_adj_42d' in features_df.columns:
            vol_adj_momentum = features_df['momentum_vol_adj_42d'].fillna(0).values
            mid_signal += 0.20 * np.clip(vol_adj_momentum, -2, 2)
        
        # Trend strength indicator
        if 'trend_strength_42d' in features_df.columns:
            trend_strength = features_df['trend_strength_42d'].fillna(0).values
            mid_signal += 0.15 * np.sign(trend_strength)
        
        # Mean reversion signal
        if 'mean_reversion_42d' in features_df.columns:
            mean_rev = features_df['mean_reversion_42d'].fillna(0).values
            mid_signal += 0.15 * np.clip(-mean_rev, -1, 1)  # Mean reversion: fade extremes
        
        # Macro trend component
        if 'macro_momentum_42d' in features_df.columns:
            macro_momentum = features_df['macro_momentum_42d'].fillna(0).values
            mid_signal += 0.10 * np.sign(macro_momentum)
        
        # Economic acceleration
        if 'econ_acceleration_42d' in features_df.columns:
            econ_accel = features_df['econ_acceleration_42d'].fillna(0).values
            mid_signal += 0.10 * np.sign(econ_accel)
        
        # Sector rotation
        if 'sector_rotation_42d' in features_df.columns:
            sector_rot = features_df['sector_rotation_42d'].fillna(0).values
            mid_signal += 0.05 * np.sign(sector_rot)
        
        # Composite mid-period signal
        if 'mid_period_composite' in features_df.columns:
            composite = features_df['mid_period_composite'].fillna(0).values
            mid_signal += 0.10 * np.clip(composite, -1, 1)
        
        return mid_signal
        
    def fit_mid_period_ensemble(self, features_df, target_returns):
        """Train the mid-period ensemble model."""
        # Define mid-period (30%-70% of data)
        total_len = len(features_df)
        mid_start = int(total_len * 0.3)
        mid_end = int(total_len * 0.7)
        mid_period_mask = np.zeros(len(features_df), dtype=bool)
        mid_period_mask[mid_start:mid_end] = True
        
        # Select features for mid-period ensemble
        mid_features = []
        feature_cols = [
            'roc_42d', 'momentum_vol_adj_42d', 'trend_strength_42d', 
            'mean_reversion_42d', 'macro_momentum_42d', 'econ_acceleration_42d',
            'sector_rotation_42d', 'mid_period_composite'
        ]
        
        for col in feature_cols:
            if col in features_df.columns:
                mid_features.append(features_df[col].fillna(0).values.reshape(-1, 1))
        
        if mid_features:
            X = np.column_stack(mid_features)
            self.mid_period_ensemble.fit(X, target_returns, mid_period_mask)
    
    def _get_mid_period_ensemble_signal(self, features_df):
        """Get mid-period ensemble prediction."""
        if not self.mid_period_ensemble.is_trained:
            return np.zeros(len(features_df))
            
        # Prepare features for prediction
        mid_features = []
        feature_cols = [
            'roc_42d', 'momentum_vol_adj_42d', 'trend_strength_42d', 
            'mean_reversion_42d', 'macro_momentum_42d', 'econ_acceleration_42d',
            'sector_rotation_42d', 'mid_period_composite'
        ]
        
        for col in feature_cols:
            if col in features_df.columns:
                mid_features.append(features_df[col].fillna(0).values.reshape(-1, 1))
        
        if mid_features and len(mid_features) == len(feature_cols):
            X = np.column_stack(mid_features)
            return self.mid_period_ensemble.predict(X)
        else:
            return np.zeros(len(features_df))
    
    def predict_sigma(self, features_df):
        """Predict volatility using rolling measures."""
        # Base volatility from rolling volatility
        if 'volatility' in features_df.columns:
            base_vol = features_df['volatility'].fillna(0.02).values  # Convert to numpy array
        else:
            base_vol = np.full(len(features_df), 0.02)
            
        # Adjust based on feature dispersion (higher dispersion = higher vol)
        feature_dispersion = np.zeros(len(features_df))
        count = 0
        for feature in ['M1', 'M13', 'P2']:
            if feature in features_df.columns:
                dispersion = features_df[feature].rolling(21).std().fillna(0.01).values  # Convert to numpy array
                feature_dispersion += dispersion
                count += 1
        
        if count > 0:
            feature_dispersion /= count
            base_vol = base_vol * (1 + 0.5 * feature_dispersion / (base_vol + 1e-8))
        
        return np.clip(base_vol, 0.005, 0.08)
        """Predict volatility using rolling measures."""
        # Base volatility from rolling volatility
        if 'volatility' in features_df.columns:
            base_vol = features_df['volatility'].fillna(0.02).values  # Convert to numpy array
        else:
            base_vol = np.full(len(features_df), 0.02)
            
        # Adjust based on feature dispersion (higher dispersion = higher vol)
        feature_dispersion = np.zeros(len(features_df))
        count = 0
        for feature in ['M1', 'M13', 'P2']:
            if feature in features_df.columns:
                dispersion = features_df[feature].rolling(21).std().fillna(0.01).values  # Convert to numpy array
                feature_dispersion += dispersion
                count += 1
        
        if count > 0:
            feature_dispersion /= count
            base_vol = base_vol * (1 + 0.5 * feature_dispersion / (base_vol + 1e-8))
        
        return np.clip(base_vol, 0.005, 0.08)

    def _compute_drawdown_scaling(self, features_df):
        """Compute position scaling based on drawdown levels."""
        if 'cumulative_returns' not in features_df.columns:
            return np.ones(len(features_df))
            
        cumulative_returns = features_df['cumulative_returns'].fillna(0).values
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-8)
        
        # Progressive scaling based on drawdown severity
        scaling = np.ones(len(drawdown))
        
        # Mild drawdown (5-10%): reduce to 80%
        mild_mask = (drawdown >= 0.05) & (drawdown < 0.10)
        scaling[mild_mask] = 0.8
        
        # Moderate drawdown (10-15%): reduce to 60%
        moderate_mask = (drawdown >= 0.10) & (drawdown < 0.15)
        scaling[moderate_mask] = 0.6
        
        # Severe drawdown (15%+): reduce to 40%
        severe_mask = drawdown >= 0.15
        scaling[severe_mask] = 0.4
        
        return scaling

    def compute_rolling_sharpe(self, returns, window=252):
        """Compute rolling Sharpe ratio for robustness validation."""
        if len(returns) < window:
            return np.full(len(returns), np.nan)
            
        rolling_mean = returns.rolling(window=window, min_periods=window//2).mean()
        rolling_std = returns.rolling(window=window, min_periods=window//2).std()
        
        # Annualized Sharpe ratio
        rolling_sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252) + 1e-8)
        
        return rolling_sharpe.fillna(0)

class ML_Correction:
    """Small ML model for corrections to rule-based signals."""
    
    def __init__(self, alpha=1.0):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha, random_state=42)
        self.is_trained = False
        
    def fit(self, X, y):
        """Fit the correction model."""
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, X):
        """Predict corrections."""
        if not self.is_trained:
            return np.zeros(len(X))
        return self.model.predict(X)

class MidPeriodEnsemble:
    """Lightweight ML model trained only on mid-period patterns."""
    
    def __init__(self, alpha=0.1):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        self.models = {
            'ridge': Ridge(alpha=alpha, random_state=42),
            'rf': RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        }
        self.is_trained = False
        self.feature_cols = []
        
    def fit(self, X, y, mid_period_mask):
        """Fit ensemble on mid-period data only."""
        if not mid_period_mask.any():
            return
            
        # Use only mid-period data for training
        X_mid = X[mid_period_mask]
        y_mid = y[mid_period_mask]
        
        if len(X_mid) < 50:  # Need minimum samples
            return
            
        # Train ensemble models
        for name, model in self.models.items():
            try:
                model.fit(X_mid, y_mid)
            except Exception as e:
                logger.warning(f"Failed to train {name} model: {e}")
                
        self.is_trained = True
        self.feature_cols = list(range(X.shape[1]))  # Store feature count
        
    def predict(self, X):
        """Predict using ensemble of models."""
        if not self.is_trained:
            return np.zeros(len(X))
            
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict with {name} model: {e}")
                predictions.append(np.zeros(len(X)))
        
        if predictions:
            # Average predictions from all models
            return np.mean(predictions, axis=0)
        else:
            return np.zeros(len(X))

def save_model(model, path: str):
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path: str):
    return joblib.load(path)