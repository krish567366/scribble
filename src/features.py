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

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features like rolling z-scores and volatility-adjusted returns."""
    derived = pd.DataFrame(index=df.index)

    # Rolling z-scores for key features (exclude forward_returns to avoid data leakage)
    key_features = ['M1', 'M13', 'P2', 'E3', 'E2', 'I2', 'S12']  # Added high-correlation features
    for col in key_features:
        if col in df.columns:
            rolling_mean = df[col].rolling(63).mean()  # ~3 months
            rolling_std = df[col].rolling(63).std()
            derived[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    # Interaction features
    if 'M1' in df.columns and 'M13' in df.columns:
        derived['M1_M13_ratio'] = df['M1'] / (df['M13'] + 1e-8)
    if 'P2' in df.columns and 'E2' in df.columns:
        derived['P2_E2_ratio'] = df['P2'] / (df['E2'] + 1e-8)

    # Volatility-adjusted returns (use price returns instead of forward_returns)
    if 'volatility' in df.columns and 'price' in df.columns:
        price_returns = df['price'].pct_change()
        derived['vol_adj_return'] = price_returns / (df['volatility'] + 1e-8)

    # Enhanced momentum indicators
    if 'price' in df.columns:
        # Rate of change - multiple timeframes
        derived['roc_5d'] = df['price'].pct_change(5)
        derived['roc_21d'] = df['price'].pct_change(21)
        derived['roc_63d'] = df['price'].pct_change(63)  # Long-term trend

        # EMA smoothing for trend features
        derived['roc_5d_ema'] = derived['roc_5d'].ewm(span=5).mean()
        derived['roc_21d_ema'] = derived['roc_21d'].ewm(span=10).mean()

        # Moving average crossovers
        ma_short = df['price'].rolling(10).mean()
        ma_long = df['price'].rolling(30).mean()
        derived['ma_crossover'] = (ma_short - ma_long) / (ma_long + 1e-8)

        # EMA of price for smoother trend
        derived['price_ema_10'] = df['price'].ewm(span=10).mean()
        derived['price_ema_30'] = df['price'].ewm(span=30).mean()
        derived['ema_crossover'] = (derived['price_ema_10'] - derived['price_ema_30']) / (derived['price_ema_30'] + 1e-8)
        
        # MACD-like indicator
        ema12 = df['price'].ewm(span=12).mean()
        ema26 = df['price'].ewm(span=26).mean()
        derived['macd'] = ema12 - ema26
        derived['macd_signal'] = derived['macd'].ewm(span=9).mean()
        derived['macd_hist'] = derived['macd'] - derived['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        derived['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma20 = df['price'].rolling(20).mean()
        std20 = df['price'].rolling(20).std()
        derived['bb_upper'] = sma20 + 2 * std20
        derived['bb_lower'] = sma20 - 2 * std20
        derived['bb_position'] = (df['price'] - sma20) / (std20 + 1e-8)
        
        # Volume indicators (if volume data exists)
        if 'volume' in df.columns:
            derived['volume_ma'] = df['volume'].rolling(20).mean()
            derived['volume_ratio'] = df['volume'] / (derived['volume_ma'] + 1e-8)

        # Medium-term trend features for mid-period performance
        # 42-day momentum (medium-term between short and long)
        derived['roc_42d'] = df['price'].pct_change(42)
        derived['roc_42d_ema'] = derived['roc_42d'].ewm(span=7).mean()

        # Medium-term moving averages
        derived['price_sma_42'] = df['price'].rolling(42).mean()
        derived['price_ema_42'] = df['price'].ewm(span=42).mean()

        # Medium-term volatility-adjusted momentum
        vol_42d = df['price'].pct_change().rolling(42).std()
        derived['momentum_vol_adj_42d'] = derived['roc_42d'] / (vol_42d + 1e-8)

        # Medium-term trend strength
        trend_21d = df['price'].rolling(21).mean()
        trend_63d = df['price'].rolling(63).mean()
        derived['trend_strength_42d'] = (trend_21d - trend_63d) / (trend_63d + 1e-8)

        # Medium-term mean reversion
        derived['mean_reversion_42d'] = (df['price'] - derived['price_sma_42']) / (df['price'].rolling(42).std() + 1e-8)

        # Macro signal combinations for mid-period
        # Combine multiple macro features with medium-term smoothing
        macro_cols = ['M1', 'M13', 'M2', 'M3'] if all(col in df.columns for col in ['M1', 'M13', 'M2', 'M3']) else []
        if macro_cols:
            macro_sum = df[macro_cols].sum(axis=1)
            derived['macro_trend_42d'] = macro_sum.rolling(42).mean()
            derived['macro_momentum_42d'] = macro_sum.pct_change(42)

        # Economic cycle indicators (medium-term)
        econ_cols = ['E1', 'E2', 'E3'] if all(col in df.columns for col in ['E1', 'E2', 'E3']) else []
        if econ_cols:
            econ_trend = df[econ_cols].mean(axis=1)
            derived['econ_cycle_42d'] = econ_trend.rolling(42).mean()
            derived['econ_acceleration_42d'] = econ_trend.pct_change(21) - econ_trend.pct_change(42)

        # Sector rotation signals (medium-term)
        sector_cols = ['S1', 'S12', 'S5'] if all(col in df.columns for col in ['S1', 'S12', 'S5']) else []
        if sector_cols:
            sector_momentum = df[sector_cols].pct_change(42).mean(axis=1)
            derived['sector_rotation_42d'] = sector_momentum

        # Composite mid-period signal
        # Combine trend, macro, and economic signals
        signal_components = []
        if 'roc_42d' in derived.columns:
            signal_components.append(derived['roc_42d'] * 0.4)
        if 'macro_momentum_42d' in derived.columns:
            signal_components.append(derived['macro_momentum_42d'] * 0.3)
        if 'econ_acceleration_42d' in derived.columns:
            signal_components.append(derived['econ_acceleration_42d'] * 0.3)

        if signal_components:
            derived['mid_period_composite'] = sum(signal_components)

    return derived

def robust_normalize_features(df: pd.DataFrame, window: int = 252) -> Tuple[pd.DataFrame, Dict]:
    """Normalize numeric features with robust statistics (median, MAD) and winsorization."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normalized = df.copy()
    params = {}

    for col in numeric_cols:
        # Use rolling median and MAD for robust scaling
        rolling_median = df[col].rolling(window).median()
        rolling_mad = (df[col] - rolling_median).abs().rolling(window).median()

        # Winsorize extreme values to prevent outlier domination
        # Cap at 5 MAD from median
        mad_factor = 5.0
        upper_bound = rolling_median + mad_factor * rolling_mad
        lower_bound = rolling_median - mad_factor * rolling_mad

        winsorized = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Robust z-score using median and MAD
        normalized[col] = (winsorized - rolling_median) / (rolling_mad + 1e-8)  # Add small epsilon to avoid division by zero

        params[col] = {
            'median': rolling_median.iloc[-1] if not rolling_median.empty else 0,
            'mad': rolling_mad.iloc[-1] if not rolling_mad.empty else 1
        }

    return normalized, params

def encode_regime(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Encode regime using KMeans on available features."""
    # Try different sets of features that might have data - prioritize high-correlation features
    feature_sets = [
        ['M1', 'M13', 'P2', 'E3', 'E2', 'I2', 'S12'],  # High-correlation macro features
        ['M1', 'M2', 'M3', 'V1', 'V2', 'V3'],  # Macro and volatility
        ['D1', 'D2', 'D3', 'E1', 'E2', 'E3'],  # Other features
        ['volatility', 'roc_5d', 'roc_21d']  # Trend and volatility features
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

    # Add derived features
    derived_features = compute_derived_features(df_feat)
    df_feat = df_feat.join(derived_features)

    # Fill NaN values created by rolling operations
    df_feat = df_feat.bfill().fillna(0)

    # Normalize numeric features (exclude target and some categorical)
    numeric_cols = [col for col in df_feat.select_dtypes(include=[np.number]).columns
                   if col not in ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate']]
    df_feat, norm_params = robust_normalize_features(df_feat[numeric_cols + ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate']], window=252)

    # Fill any remaining NaNs
    df_feat = df_feat.fillna(0)

    # Encode regime
    df_feat = encode_regime(df_feat)
    
    # Ensure no duplicate indices
    if df_feat.index.has_duplicates:
        logger.warning("Duplicate indices in features, keeping first occurrence")
        df_feat = df_feat[~df_feat.index.duplicated(keep='first')]

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