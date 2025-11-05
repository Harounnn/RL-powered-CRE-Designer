# Week 3 — Day 4 Summary

## Experiments Recap

### 1. **Hybrid and SVD experiments (C1–C2)**
We compared several representation methods:
- **k=7 and hybrid (k=6+k7)** models produced only marginal improvements over k=6 alone.
- **TruncatedSVD (200 & 500 components)** failed to improve performance, slightly reducing Spearman correlation.
  - This suggested that dimensionality reduction blurred informative motif patterns captured by sparse features.

### 2. **CNN experiment**
We trained a compact 1D-CNN using one-hot encoding to capture positional motif information.
- Training on CPU took ~1.3 hours.
- **Best validation Spearman:** 0.281  
- **Test Spearman:** 0.265  
- CNN underperformed compared to the LightGBM baseline (~0.33 Spearman).

### Interpretation
The CNN’s underperformance likely stems from:
- CPU-only training and limited epochs,
- Architectural under-tuning,
- Label noise in low-coverage tiles,
- Sparse motifs being better handled by k-mer features.

---

## Decision

We **selected the LightGBM k=6 raw-count model** as our **final surrogate predictor** for promoter activity due to:
- Stable convergence,
- Best Spearman correlation (~0.33),
- Simplicity and interpretability (motif-level feature importance).

This model will serve as the foundation for the **RL reward function**, where its normalized predictions (`predictor_score(seq)`) will represent promoter strength.

---


## Takeaway

Our exploration confirmed that **non-positional k-mer models** remain a strong baseline for CRE activity prediction, while CNNs require more resources and tuning to outperform them.  
This establishes a **solid surrogate model** for the upcoming **reinforcement learning phase**, where we’ll shift focus from prediction to **sequence optimization**.

---
