import pandas as pd
import numpy as np

# Load features
df = pd.read_parquet('cache/features.parquet')
print('Feature statistics:')
stats = df.describe().T[['mean', 'std', 'min', 'max']].round(4)
print(stats)

print('\nFeatures with extreme negative values:')
extreme_cols = ['M1', 'M13', 'P2']
for col in extreme_cols:
    if col in df.columns:
        print(f'{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, std={df[col].std():.4f}')

print('\nFeatures with very small magnitude:')
small_features = []
for col in df.columns:
    if df[col].std() < 0.01 and abs(df[col].mean()) < 0.01:
        small_features.append(col)

print(f'Small magnitude features: {small_features[:10]}')  # Show first 10

print('\nHigh variance features (std > 1):')
high_var_features = []
for col in df.columns:
    if df[col].std() > 1:
        high_var_features.append(col)

print(f'High variance features: {high_var_features[:10]}')  # Show first 10