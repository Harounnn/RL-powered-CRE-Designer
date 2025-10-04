"""
Process SHARPR counts files:
- Aggregates Tags -> sequence counts per file
- Pairs mRNA (RNA) files with plasmid/DNA files
- Sums replicates (if multiple)
- Computes log2((RNA+1)/(DNA+1)) as activity
- Writes processed CSV: processed_sharpr_seq_activity.csv
"""

import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict

DATA_DIR = "."  
OUT_DIR = "../../processed/sharpr"  
os.makedirs(OUT_DIR, exist_ok=True)

def read_counts(path):
    df = pd.read_csv(path, sep='\t', header=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

# find all .txt/.counts files in folder
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.*")))
# filter likely counts files by looking for 'counts' or 'mRNA' in filename or by columns
candidates = []
for f in files:
    if f.lower().endswith(('.txt','.counts','.tsv','.csv')):
        candidates.append(f)

print("Found candidate files:", len(candidates))
for f in candidates:
    print("  ", os.path.basename(f))

# Simple heuristic to classify RNA vs DNA files by filename
rna_files = [f for f in candidates if ('mRNA' in os.path.basename(f)) or ('mrna' in os.path.basename(f)) or ('rna' in os.path.basename(f) and 'plasmid' not in os.path.basename(f).lower())]
dna_files = [f for f in candidates if ('plasmid' in os.path.basename(f).lower()) or ('dna' in os.path.basename(f).lower()) or ('plasmid' in os.path.basename(f))]

print("\nInferred RNA files:", [os.path.basename(x) for x in rna_files][:10])
print("Inferred DNA/plasmid files:", [os.path.basename(x) for x in dna_files][:10])

# Function to aggregate a counts dataframe by Sequence
def agg_by_sequence(df):
    cols = [c.lower() for c in df.columns]
    seq_col = None
    for name in ['sequence','seq','construct','oligo','sequence_string','sequence.full']:
        if name in cols:
            seq_col = df.columns[cols.index(name)]
            break
    if seq_col is None:
        # fallback: try to find a column whose values are only ACGTN
        for c in df.columns:
            sample = str(df[c].iloc[0])
            if set(sample.upper()) <= set("ACGTN"):
                seq_col = c
                break
    if seq_col is None:
        raise ValueError("Could not detect sequence column in file. Inspect columns: " + ",".join(df.columns))

    # find count column
    count_col = None
    for name in ['counts','count','read_count','reads','tag_count','value','score']:
        if name in cols:
            count_col = df.columns[cols.index(name)]
            break
    if count_col is None:
        # fallback: find first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                count_col = c
                break
    if count_col is None:
        raise ValueError("Could not detect count column. Inspect columns: " + ",".join(df.columns))

    df[count_col] = pd.to_numeric(df[count_col], errors='coerce').fillna(0).astype(int)
    agg = df.groupby(seq_col)[count_col].sum().reset_index().rename(columns={seq_col:'sequence', count_col:'counts'})
    return agg

# Aggregate all RNA files into a single RNA count per sequence (sum across replicates/files)
rna_agg_list = []
for f in rna_files:
    try:
        df = read_counts(f)
        ag = agg_by_sequence(df)
        ag = ag.rename(columns={'counts': os.path.basename(f)})  # temporarily keep file-specific counts
        rna_agg_list.append(ag)
    except Exception as e:
        print("Skipping file", f, "due to error:", e)

# merge RNA replicates by sequence (outer join then sum)
if not rna_agg_list:
    raise RuntimeError("No RNA files aggregated. Check file detection heuristics.")
rna_merged = rna_agg_list[0]
for other in rna_agg_list[1:]:
    rna_merged = pd.merge(rna_merged, other, on='sequence', how='outer')
rna_merged = rna_merged.fillna(0)
# sum across all RNA columns to create rna_total
rna_cols = [c for c in rna_merged.columns if c != 'sequence']
rna_merged['rna_total'] = rna_merged[rna_cols].sum(axis=1).astype(int)
rna_df = rna_merged[['sequence','rna_total']].copy()
print("\nRNA aggregated: sequences:", len(rna_df))

# Aggregate DNA/plasmid files if found; if not, try to infer single DNA file in folder
dna_df = None
if dna_files:
    dna_agg_list = []
    for f in dna_files:
        try:
            df = read_counts(f)
            ag = agg_by_sequence(df)
            ag = ag.rename(columns={'counts': os.path.basename(f)})
            dna_agg_list.append(ag)
        except Exception as e:
            print("Skipping DNA file", f, "due to error:", e)
    if dna_agg_list:
        dna_merged = dna_agg_list[0]
        for other in dna_agg_list[1:]:
            dna_merged = pd.merge(dna_merged, other, on='sequence', how='outer')
        dna_merged = dna_merged.fillna(0)
        dna_cols = [c for c in dna_merged.columns if c != 'sequence']
        dna_merged['dna_total'] = dna_merged[dna_cols].sum(axis=1).astype(int)
        dna_df = dna_merged[['sequence','dna_total']].copy()
        print("DNA aggregated: sequences:", len(dna_df))
else:
    print("No DNA/plasmid files auto-detected. If the experiment has a plasmid file, add it to the folder or update heuristics.")

# Merge RNA and DNA counts (inner join on sequence)
if dna_df is not None:
    merged = pd.merge(rna_df, dna_df, on='sequence', how='inner')
else:
    # If no DNA, just keep RNA counts and set dna_total to 0 (we'll still compute log ratio with +1)
    merged = rna_df.copy()
    merged['dna_total'] = 0

# compute log2 ratio
merged['rna_plus1'] = merged['rna_total'] + 1
merged['dna_plus1'] = merged['dna_total'] + 1
merged['log2_ratio'] = np.log2(merged['rna_plus1'] / merged['dna_plus1'])

# basic filters: remove sequences with extremely low coverage if desired


# Save processed table
out_csv = os.path.join(OUT_DIR, "processed_sharpr_sequence_activity.csv")
merged.to_csv(out_csv, index=False)
print("Wrote processed table to", out_csv)
print("Summary stats for log2_ratio:")
print(merged['log2_ratio'].describe())

# Save small sample for preview
merged.head(20).to_csv(os.path.join(OUT_DIR, "processed_sharpr_sample_head20.csv"), index=False)
print("Saved sample head20.")
