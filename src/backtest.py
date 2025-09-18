import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from .utils import save_json

logger = logging.getLogger(__name__)

def compute_weights(mu: np.ndarray, sigma: np.ndarray, k: float, variant: str = 'conservative', df: pd.DataFrame = None) -> np.ndarray:
    """Compute position weights with variant adjustments."""
    base_weights = np.clip(mu / sigma * k, 0, 2)
    
    if variant == 'conservative':
        # Lower leverage
        weights = base_weights * 0.8
    elif variant == 'aggressive':
        # Higher leverage, less clipping
        weights = np.clip(mu / sigma * k * 1.2, 0, 2)
    elif variant == 'mid':
        # Regime-dependent: reduce in high vol regimes
        if df is not None and 'regime_0' in df.columns:
            # Assume regime_0 is high vol
            high_vol_mask = df['regime_0'] == 1
            weights = base_weights.copy()
            weights[high_vol_mask] *= 0.7  # Reduce in high vol
        else:
            weights = base_weights
    else:
        weights = base_weights
    
    return np.clip(weights, 0, 2)

def apply_ema_smoothing(weights: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply exponential moving average smoothing to weights."""
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
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized

def calibrate_k(returns: pd.Series, mu: np.ndarray, sigma: np.ndarray, target_vol: float = 0.12, variant: str = 'conservative', df: pd.DataFrame = None) -> float:
    """Calibrate k to meet target volatility."""
    def objective(k):
        weights = compute_weights(mu, sigma, k, variant, df)
        port_returns = simulate_portfolio(returns, weights)
        vol = port_returns.std() * np.sqrt(252)
        return abs(vol - target_vol)
    
    # Simple grid search
    k_values = np.linspace(0.1, 5.0, 50)
    best_k = min(k_values, key=objective)
    return best_k

def stress_test(returns: pd.Series, weights: np.ndarray, periods: List[Tuple[str, str]]) -> Dict:
    """Stress test on specific periods."""
    results = {}
    for name, (start, end) in periods:
        mask = (returns.index >= start) & (returns.index <= end)
        port_returns = simulate_portfolio(returns[mask], weights[mask])
        sharpe = calculate_sharpe(port_returns)
        drawdown = (port_returns.cumsum() - port_returns.cumsum().cummax()).min()
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