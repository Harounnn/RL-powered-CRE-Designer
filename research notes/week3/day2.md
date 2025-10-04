## Data Preprocessing + k-mer baseline
- Loaded the SHARPR processed sequence table and computed per-sequence `log2(RNA/DNA)` activity.  
- Filtered out low-coverage sequences (`rna_total + dna_total < 10`).  
  - **Before filter:** 242,276 sequences  
  - **After filter:** 226,538 sequences (≈15.7k removed)  
- Built **overlapping 6-mer** count features (CountVectorizer).  
  - Feature matrix shape: **(226,538, 4,094)**  
  - Vocabulary size: **4,094 unique 6-mers** (max possible is 4⁶ = 4,096).  
  - Matrix density ≈ **0.0327** → about **3.3%** of entries are non-zero (sparse matrix).  
- Trained a quick sparse linear baseline (SGDRegressor) and evaluated:
  - **MAE:** 0.8832 (in log₂ units)  
  - **Spearman r:** 0.2567 (rank correlation — modest)
---

## Interpretation 
- **Coverage:** the sum of RNA + DNA reads for a sequence. Low coverage means noisy measurements; we removed low-coverage items to reduce noise.
- **k-mer features:** each sequence was converted into counts of every overlapping 6-base substring (a compact, motif-aware representation).
- **Sparsity / density:** most sequences don't contain most k-mers, so the feature matrix is sparse (efficient to store / compute on).
- **MAE = 0.88 (log₂):** on average predictions differ by ~0.88 log₂ units — roughly a ~1.85× fold error in RNA/DNA scale.  
- **Spearman ≈ 0.26:** the model captures some ranking signal (sequences it scores high tend to be truly higher), but performance is modest — there is room to improve the surrogate.

The top positive k-mers (highest positive weights) are candidates for motifs associated with higher activity in this library. This supports the idea that motif-level signals are present and that k-mer features are informative.

---

## Relation to past & next work
- **From Week 1–2:** we decided on biologically meaningful reward terms (motifs, GC, predictor) and chose overlapping 6-mers as our default encoding.  
- **Today:** validated that 6-mer features produce an interpretable, sparse feature set and yield a usable baseline predictor (modest but non-zero Spearman).  
- **Next:** we will iterate on the predictor until it is robust enough to supply stable rewards for the RL agent:
  - improve model performance (better Spearman/MAE),
  - examine feature / motif signals,
  - save a final predictor that exposes `predictor_score(seq)` for the RL reward.