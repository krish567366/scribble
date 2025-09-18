import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Any
import matplotlib.pyplot as plt
from .utils import save_json

logger = logging.getLogger(__name__)

def compute_dynamic_k(mu: np.ndarray, sigma: np.ndarray, base_k: float, df: pd.DataFrame = None) -> np.ndarray:
    """Compute dynamic k based on predicted return magnitude, volatility, and regime with enhanced mid-period handling."""
    k_values = np.full(len(mu), base_k)

    if df is not None:
        # Adjust k based on return magnitude (higher confidence -> higher k)
        return_magnitude = np.abs(mu)
        confidence_factor = np.clip(return_magnitude / (np.std(return_magnitude) + 1e-8), 0.5, 2.0)
        k_values *= confidence_factor

        # Enhanced volatility-based regime detection with mid-period focus
        if 'volatility' in df.columns:
            vol_series = df['volatility']

            # Multi-timeframe volatility assessment
            vol_5d = vol_series.rolling(5, min_periods=1).mean().fillna(vol_series.mean())
            vol_21d = vol_series.rolling(21, min_periods=1).mean().fillna(vol_series.mean())
            vol_63d = vol_series.rolling(63, min_periods=1).mean().fillna(vol_series.mean())

            # Volatility regime classification
            vol_regime = np.zeros(len(df))

            # Low volatility: below 20th percentile of 21d vol
            low_vol_mask = vol_21d <= vol_21d.quantile(0.2)
            vol_regime[low_vol_mask] = 0

            # Normal volatility: 20th-60th percentile
            normal_vol_mask = (vol_21d > vol_21d.quantile(0.2)) & (vol_21d <= vol_21d.quantile(0.6))
            vol_regime[normal_vol_mask] = 1

            # High volatility: 60th-80th percentile
            high_vol_mask = (vol_21d > vol_21d.quantile(0.6)) & (vol_21d <= vol_21d.quantile(0.8))
            vol_regime[high_vol_mask] = 2

            # Extreme volatility: above 80th percentile
            extreme_vol_mask = vol_21d > vol_21d.quantile(0.8)
            vol_regime[extreme_vol_mask] = 3

            # Apply regime-specific k scaling with mid-period emphasis
            k_values[vol_regime == 0] *= 1.3      # Low vol: increase k to 2.6
            k_values[vol_regime == 1] *= 1.0      # Normal vol: keep base k = 2.0
            k_values[vol_regime == 2] *= 0.5      # High vol: reduce k to 1.0 (more conservative)
            k_values[vol_regime == 3] *= 0.2      # Extreme vol: reduce k to 0.4

        # Enhanced regime adjustment based on clustering with mid-period focus
        regime_cols = [col for col in df.columns if col.startswith('regime_')]
        if regime_cols:
            # Calculate regime volatility score
            regime_volatility = np.zeros(len(df))
            for i, col in enumerate(regime_cols):
                regime_volatility += df[col].values * (i + 1)  # Higher regime numbers = more volatile

            # Mid-period specific regime adjustment (assuming mid-period is indices ~30%-70%)
            total_len = len(df)
            mid_start = int(total_len * 0.3)
            mid_end = int(total_len * 0.7)
            mid_period_mask = (np.arange(len(df)) >= mid_start) & (np.arange(len(df)) < mid_end)

            # More conservative in mid-period for high-volatility regimes
            mid_high_vol_mask = mid_period_mask & (regime_volatility > 2)
            k_values[mid_high_vol_mask] *= 0.7  # Additional 30% reduction in mid-period high vol

            # Less conservative in low-volatility regimes
            mid_low_vol_mask = mid_period_mask & (regime_volatility <= 1)
            k_values[mid_low_vol_mask] *= 1.1  # Slight increase in mid-period low vol

        # Enhanced drawdown protection with multiple thresholds
        if len(df) > 20:
            if 'price' in df.columns:
                price_returns = df['price'].pct_change().fillna(0)
                cum_returns = (1 + price_returns).cumprod()

                # Multiple drawdown horizons
                dd_21d = ((cum_returns - cum_returns.rolling(21, min_periods=1).max()) / cum_returns.rolling(21, min_periods=1).max())
                dd_63d = ((cum_returns - cum_returns.rolling(63, min_periods=1).max()) / cum_returns.rolling(63, min_periods=1).max())
                dd_126d = ((cum_returns - cum_returns.rolling(126, min_periods=1).max()) / cum_returns.rolling(126, min_periods=1).max())

                # Progressive drawdown protection
                mild_dd_mask = (dd_21d < -0.05) | (dd_63d < -0.08) | (dd_126d < -0.12)
                moderate_dd_mask = (dd_21d < -0.10) | (dd_63d < -0.15) | (dd_126d < -0.20)
                severe_dd_mask = (dd_21d < -0.15) | (dd_63d < -0.25) | (dd_126d < -0.30)

                k_values[mild_dd_mask] *= 0.85      # 15% reduction
                k_values[moderate_dd_mask] *= 0.7   # 30% reduction
                k_values[severe_dd_mask] *= 0.5     # 50% reduction

        # Signal quality adjustment
        signal_quality = np.abs(mu) / (sigma + 1e-8)
        low_quality_mask = signal_quality < np.percentile(signal_quality, 20)
        k_values[low_quality_mask] *= 0.8  # Reduce exposure for low-quality signals

    return np.clip(k_values, 0.0, base_k * 2.0)

def compute_weights(mu: np.ndarray, sigma: np.ndarray, k: float, variant: str = 'conservative', df: pd.DataFrame = None, config: Dict[str, Any] = None) -> np.ndarray:
    """Compute position weights with variant adjustments and dynamic k scaling."""
    if config is None:
        config = {}
    regime_sensitivity = config.get('model', {}).get('regime_sensitivity', 0.7)

    # Apply dynamic k scaling
    dynamic_k = compute_dynamic_k(mu, sigma, k, df)

    # Compute base weights with dynamic k
    base_weights = np.clip(mu / sigma * dynamic_k, 0, 2)

    if variant == 'conservative':
        # Lower leverage
        weights = base_weights * 0.8
    elif variant == 'aggressive':
        # Higher leverage, less clipping
        weights = np.clip(mu / sigma * dynamic_k * 1.2, 0, 2)
    elif variant == 'mid':
        # Regime-dependent: reduce in high vol regimes
        if df is not None and 'regime_0' in df.columns:
            # Assume regime_0 is high vol
            high_vol_mask = df['regime_0'].values == 1  # Convert to numpy array
            weights = base_weights.copy()
            weights[high_vol_mask] *= regime_sensitivity  # Use config value
        else:
            weights = base_weights
    else:
        weights = base_weights

    return np.clip(weights, 0, 2)

def apply_ema_smoothing(weights: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential moving average smoothing to weights."""
    # Ensure weights is 1D
    weights = np.asarray(weights).flatten()
    
    # Handle NaN values
    if np.isnan(weights).any():
        logger.warning("NaN values found in weights, filling with 0")
        weights = np.nan_to_num(weights, nan=0.0)
    
    smoothed = np.zeros_like(weights)
    smoothed[0] = weights[0]
    for i in range(1, len(weights)):
        smoothed[i] = alpha * weights[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def simulate_portfolio(returns: pd.Series, weights: np.ndarray) -> pd.Series:
    """Simulate portfolio returns."""
    portfolio_returns = returns * weights
    return portfolio_returns

def calculate_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free
    
    # Handle NaN values
    if excess_returns.isna().all():
        return 0.0
    
    # Remove NaN values for calculation
    excess_returns = excess_returns.dropna()
    
    if len(excess_returns) == 0 or excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized

def calibrate_k(returns: pd.Series, mu: np.ndarray, sigma: np.ndarray, target_vol: float = 0.12, variant: str = 'conservative', df: pd.DataFrame = None, config: Dict[str, Any] = None) -> float:
    """Calibrate k to meet target volatility."""
    def objective(k):
        weights = compute_weights(mu, sigma, k, variant, df, config)
        port_returns = simulate_portfolio(returns, weights)
        vol = port_returns.std() * np.sqrt(252)
        return abs(vol - target_vol)
    
    # Simple grid search
    k_values = np.linspace(0.1, 5.0, 50)
    best_k = min(k_values, key=objective)
    return best_k

def stress_test(returns: pd.Series, weights: np.ndarray, periods: List[Tuple[str, int, int]]) -> Dict:
    """Stress test on specific periods."""
    results = {}
    for name, start, end in periods:
        mask = (returns.index >= start) & (returns.index <= end)
        port_returns = simulate_portfolio(returns[mask], weights[mask])
        sharpe = calculate_sharpe(port_returns)
        
        # Correct drawdown calculation
        if len(port_returns) > 0:
            cumulative_value = (1 + port_returns).cumprod()
            drawdown_series = (cumulative_value - cumulative_value.cummax()) / cumulative_value.cummax()
            drawdown = drawdown_series.min()
        else:
            drawdown = 0.0
            
        results[name] = {'sharpe': sharpe, 'max_drawdown': drawdown}
    return results

def plot_performance(returns: pd.Series, weights: np.ndarray, path: str):
    """Plot cumulative returns and rolling Sharpe."""
    port_returns = simulate_portfolio(returns, weights)
    cum_returns = (1 + port_returns).cumprod()
    rolling_sharpe = port_returns.rolling(252).apply(calculate_sharpe)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(cum_returns)
    ax1.set_title('Cumulative Returns')
    ax2.plot(rolling_sharpe)
    ax2.set_title('Rolling Sharpe')
    plt.savefig(path)
    plt.close()
    logger.info(f"Plot saved to {path}")