## Decision on data
We chose **SHARPR-MPRA (GEO: GSE71279)** as the primary dataset for training our continuous surrogate predictor.  
Rationale (short): SHARPR provides **human cell-type** MPRA measurements (continuous per-tile activity), a good balance of biological relevance and dataset size, and is widely used as a benchmark for sequence→activity modeling. This matches our goal of training a continuous predictor (`predictor_score(seq)`) the RL agent will optimize.

---

## Why SHARPR fits our project
- **Directly relevant:** Human promoter/regulatory activity → no species-transfer issues for a human-focused CRE designer.  
- **Continuous labels:** We can train a regression predictor which maps directly to RL reward design.  
- **Size & diversity:** Enough examples to train a robust surrogate without requiring massive compute.