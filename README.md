# Forecasting Pipeline

This project implements a production-ready forecasting pipeline for the Kaggle competition, focusing on risk-adjusted excess returns prediction.

## Project Structure

- `nbk/`: Jupyter notebooks
- `scripts/`: CLI scripts
- `src/`: Modular source code
- `artifacts/`: Trained models and reports
- `cache/`: Precomputed features
- `submissions/`: Submission files
- `tests/`: Unit tests

## Quick Start

### Prerequisites

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt` (create this file with pandas, numpy, scikit-learn, lightgbm, etc.)

### Running the Pipeline

#### CLI

```bash
python scripts/run_pipeline.py \
  --config configs/baseline.yaml \
  --data-root /data \
  --rebuild-cache \
  --use-gpu \
  --variant conservative \
  --n-folds 5 \
  --train-window 2520 \
  --val-window 252
```

#### Config Files

Use YAML config files in `configs/` to specify parameters:

- `configs/baseline.yaml`: Conservative baseline configuration
- `configs/variant1.yaml`: Aggressive variant with different parameters

Config parameters include model hyperparameters, leverage settings, and backtest parameters.

Open `nbk/forecast_pipeline.ipynb` and run all cells. Ensure data is in `/data`.

## Configuration

- `--rebuild-cache`: Recompute features and data cache
- `--use-gpu`: Enable GPU for LightGBM
- `--variant`: `conservative`, `aggressive`, or `mid` (affects leverage; mid uses regime-dependent weights)
- `--n-folds`: Number of CV folds
- `--train-window`: Training window size (days)
- `--val-window`: Validation window size (days)

## Runtime Optimization

To stay under 8 hours:

- Reduce `--n-folds` to 3
- Disable `--use-gpu` if no GPU
- Use precomputed features (avoid `--rebuild-cache`)
- Limit temporal NN with `--use-temporal-nn` flag (not implemented yet)

## Output

- `submissions/submission.csv`: Weights for submission
- `artifacts/run_report.json`: Metrics and config
- `artifacts/performance.png`: Backtest plots

## Testing

Run tests: `python -m pytest tests/`

## Notes

- Assumes data has columns: `date`, `price`, `market_forward_excess_returns`, etc.
- Adjust column names in code as needed.
- Variants: `conservative` (lower leverage), `aggressive` (higher leverage), `mid` (regime-dependent weights)
- EMA smoothing is applied to reduce noise in weights
- For production, add more robust error handling and logging.
