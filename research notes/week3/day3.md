# LightGBM baseline 

## Outcome
Trained a LightGBM regressor on overlapping **6-mer** features extracted from SHARPR; performance improved over the linear baseline (MAE ↓, Spearman ↑).

---

## Key numbers & artifacts
- **Data after QC (coverage ≥ 10):** 226,538 sequences (from 242,276).  
- **Features:** overlapping 6-mers → **vocab = 4,094**; feature matrix density ≈ **3.27%** (sparse).  
- **Model:** LightGBM booster saved at `models\predictor_sharpr_lightgbm.txt`.  
- **Test performance:**  
  - **MAE:** **0.810316** (log₂ units)  
  - **Spearman r:** **0.328307** (rank correlation)
- **Reports saved:** `data/processed/sharpr/reports/day3_lightgbm_report.txt` and `top_kmers_pos.txt` / `top_kmers_neg.txt`.

---

## Top signals (short)
- **Top positive 6-mers** (sample): `TTCCGG`, `CTTCCG`, `CCGGAA`, `GGGCGG`, `CGGAAG`, ... — many are **GC-rich** and repeatedly appear among the top positives.  
- **Top negative 6-mers** (sample): `CACCTG`, `ATTCTT`, `AAATCA`, `TTCATT`, ... — many are **AT-rich**.

*Interpretation:* the model finds strong GC-rich sequence patterns that associate with higher activity in this library; AT-rich patterns tend to correlate negatively.

---

## Simple Explanations
- **Coverage:** sum of RNA + DNA reads for a sequence. Low coverage → noisy activity estimates. We filtered sequences with coverage < 10 to reduce noise.  
- **k-mer (6-mer):** all overlapping length-6 substrings of a sequence; a compact motif-aware representation.  
- **Sparse matrix:** most sequences do not contain most possible 6-mers, so the feature matrix stores only nonzero counts (memory efficient).  
- **MAE (Mean Absolute Error):** average absolute difference between predicted and true `log2(RNA/DNA)` values. Lower is better; 0.81 means on average predictions are off by ~0.81 log₂ units.  
- **Spearman r:** rank correlation between predicted and true values (0=no relation, 1=perfect ranking). Our **~0.33** is a modest but useful ranking signal for RL reward purposes.  
- **Feature correlation / importance:** per-kmer Pearson correlations and LightGBM importances tell us which 6-mers most strongly associate (positively/negatively) with activity.

---

## Takeaways
- The LightGBM model **improves** over the earlier sparse linear baseline (MAE 0.88 → 0.81; Spearman 0.26 → 0.33).  
- The predictor captures **meaningful motif-level signals** (GC-rich motifs), so k-6 features are informative for promoter activity in this dataset.  
- Spearman ≈ 0.33 is **useful but not perfect** — good enough to bootstrap RL experiments, but better surrogate accuracy would improve RL sample efficiency and solution quality.