import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import pyarrow as pa
import pyarrow.parquet as pq
from .utils import save_json

logger = logging.getLogger(__name__)

def compute_rolling_returns(df: pd.DataFrame, windows: List[int] = [1, 5, 21]) -> pd.DataFrame:
    """Compute rolling returns for given windows."""
    returns = {}
    for w in windows:
        returns[f'return_{w}d'] = df['price'].pct_change(w, fill_method=None)
    return pd.DataFrame(returns, index=df.index)

def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Compute rolling volatility."""
    return df['price'].pct_change(fill_method=None).rolling(window).std()

def normalize_features(df: pd.DataFrame, window: int = 252) -> Tuple[pd.DataFrame, Dict]:
    """Normalize numeric features with rolling z-score."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    normalized = df.copy()
    params = {}
    for col in numeric_cols:
        rolling_mean = df[col].rolling(window).mean()
        rolling_std = df[col].rolling(window).std()
        normalized[col] = (df[col] - rolling_mean) / rolling_std
        params[col] = {'mean': rolling_mean.iloc[-1], 'std': rolling_std.iloc[-1]}
    return normalized, params

def encode_regime(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Encode regime using KMeans on available features."""
    # Try different sets of features that might have data
    feature_sets = [
        ['M1', 'M2', 'M3', 'V1', 'V2', 'V3'],  # Macro and volatility
        ['D1', 'D2', 'D3', 'E1', 'E2', 'E3'],  # Other features
        ['volatility', 'return_1d', 'return_5d']  # Computed features
    ]
    
    for features in feature_sets:
        available_features = [f for f in features if f in df.columns]
        if available_features:
            X = df[available_features].dropna()
            if len(X) > n_clusters * 2:  # Need enough data for clustering
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    regimes = kmeans.fit_predict(X)
                    
                    # Create a Series with the same index as X, then reindex to match df
                    regime_series = pd.Series(regimes, index=X.index)
                    df['regime'] = regime_series.reindex(df.index)
                    
                    # Fill NaN regimes with the most common regime
                    if df['regime'].isna().any():
                        most_common_regime = df['regime'].mode().iloc[0] if not df['regime'].mode().empty else 0
                        df['regime'] = df['regime'].fillna(most_common_regime).astype(int)
                    
                    df = pd.get_dummies(df, columns=['regime'], prefix='regime')
                    logger.info(f"Regime encoding successful using features: {available_features}")
                    return df
                except Exception as e:
                    logger.warning(f"Clustering failed with features {available_features}: {e}")
                    continue
    
    # If no clustering worked, create a simple regime based on volatility
    if 'volatility' in df.columns:
        df['regime'] = (df['volatility'] > df['volatility'].quantile(0.7)).astype(int)
        df = pd.get_dummies(df, columns=['regime'], prefix='regime')
        logger.info("Using simple volatility-based regime encoding")
    else:
        logger.warning("No suitable features found for regime encoding, skipping")
    
    return df

def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create all features."""
    df_feat = df.copy()
    
    # Compute rolling returns and volatility
    df_feat = df_feat.join(compute_rolling_returns(df_feat))
    df_feat['volatility'] = compute_volatility(df_feat)
    
    # Normalize numeric features (exclude target and some categorical)
    numeric_cols = [col for col in df_feat.select_dtypes(include=[np.number]).columns 
                   if col not in ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate']]
    df_feat, norm_params = normalize_features(df_feat[numeric_cols + ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate']], window=252)
    
    # Encode regime
    df_feat = encode_regime(df_feat)
    
    return df_feat, norm_params

def save_features(df: pd.DataFrame, path: str) -> None:
    """Save features to Parquet."""
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    logger.info(f"Features saved to {path}")

def load_features(path: str) -> pd.DataFrame:
    """Load features from Parquet."""
    table = pq.read_table(path)
    return table.to_pandas()