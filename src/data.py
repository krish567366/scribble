import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Tuple
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pyarrow as pa
import pyarrow.parquet as pq
from .utils import setup_logging

logger = logging.getLogger(__name__)

def load_raw_data(data_root: str) -> pd.DataFrame:
    """Load raw data from /data directory."""
    # Assume data is in CSV or Parquet; adjust as needed
    data_path = os.path.join(data_root, 'data.csv')  # or 'data.parquet'
    if os.path.exists(data_path):
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, parse_dates=['date'])  # assume 'date' column
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported data format")
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Set index if needed
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    logger.info(f"Loaded data with shape {df.shape}")
    return df

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: forward-fill for time series, iterative imputer for systematic."""
    df_imputed = df.copy()
    
    # Forward-fill for time series gaps
    df_imputed = df_imputed.fillna(method='ffill')
    
    # For remaining NaNs, use iterative imputer (simple version of SoftImpute)
    imputer = IterativeImputer(random_state=42, max_iter=10)
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    
    logger.info("Imputation completed")
    return df_imputed

def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to Parquet."""
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    logger.info(f"Saved to {path}")

def load_from_parquet(path: str) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    table = pq.read_table(path)
    return table.to_pandas()

def get_cached_data(cache_path: str, rebuild: bool, data_root: str) -> pd.DataFrame:
    """Get data from cache or rebuild."""
    if not rebuild and os.path.exists(cache_path):
        logger.info("Loading from cache")
        return load_from_parquet(cache_path)
    else:
        logger.info("Rebuilding cache")
        df = load_raw_data(data_root)
        df = impute_data(df)
        save_to_parquet(df, cache_path)
        return df