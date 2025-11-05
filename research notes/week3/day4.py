import pandas as pd
import numpy as np

### Analyze coverage distribution and filtering impact

# Load your processed file
df = pd.read_csv("data/processed/sharpr/processed_sharpr_sequence_activity.csv")

# Compute total coverage per sequence
df['coverage'] = df['rna_total'].fillna(0).astype(int) + df['dna_total'].fillna(0).astype(int)

# Bin the DNA coverage (log scale bins)
bins = [-1, 0, 1, 2, 4, 9, 19, 49, 199, 999999]
labels = ['0', '1', '2', '3-4', '5-9', '10-19', '20-49', '50-199', '200+']
df['dna_bin'] = pd.cut(df['dna_total'], bins=bins, labels=labels)

# Aggregate stats by bin
grp = df.groupby('dna_bin')['log2_ratio'].agg(['count', 'mean', 'median', 'std']).reset_index()

print("Coverage bins summary:")
print(grp.to_string(index=False))

# Check filtering impact
before = len(df)
after = df[df['coverage'] >= 10].shape[0]
print(f"\nTotal sequences: {before}, after coverage>=10: {after}, removed: {before - after} ({(before - after) / before:.1%})")

