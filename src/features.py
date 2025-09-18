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
        returns[f'return_{w}d'] = df['price'].pct_change(w)  # assume 'price' column
    return pd.DataFrame(returns, index=df.index)

def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Compute rolling volatility."""
    return df['price'].pct_change().rolling(window).std()

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
    """Encode regime using KMeans on macro + vol features."""
    features = ['macro_feature', 'volatility']  # assume these columns exist
    if not all(f in df.columns for f in features):
        logger.warning("Regime features not found, skipping")
        return df
    X = df[features].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = kmeans.fit_predict(X)
    df['regime'] = pd.Series(regimes, index=X.index)
    return pd.get_dummies(df, columns=['regime'], prefix='regime')

def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create all features."""
    df_feat = df.copy()
    df_feat = df_feat.join(compute_rolling_returns(df_feat))
    df_feat['volatility'] = compute_volatility(df_feat)
    df_feat, norm_params = normalize_features(df_feat)
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