# Week 3 — Day 5 (W3D5) — Final Summary

## 1) What we finalized today
- **Surrogate predictor:** LightGBM trained on raw overlapping **6-mer** counts (best params from tuning).  
  - Saved model: `models/predictor_sharpr_lightgbm.txt`
  - Feature vocab: `data/processed/sharpr/vocab_k6.json`
  - Feature matrix: `data/processed/sharpr/X_k6.npz`
  - Labels: `data/processed/sharpr/y.npy`

- **Percentile calibrator (predictor → [0,1]):**  
  - Computed raw predictions across the training set and saved:
    - `models/predictor_raw_preds.npy` (raw predictions)
    - `models/predictor_percentiles.npy` (sorted training preds for mapping → percentile)

- **Production predictor wrapper:** `src/predictor.py` (class `CREPredictor`)  
  - Robust k-mer tokenization (handles both list/dict vocabs), manual k-mer count vectorization, LightGBM inference, and percentile mapping function `percentile_score(seq)`.

---

## 2) Predictor validation (sanity)
- Random sequences produce **wide-ranging** percentile scores (`min ~0.002` → `max ~0.98`), mean ≈ `0.58`.  
- Ranking test produced diverse percentiles (range ~0.10 → 0.93), showing the surrogate has useful rank signal.  
- Insertion of a canonical motif (TATA) did **not always** increase score (dependent on sequence context), which indicates the predictor learned **rich multi-kmer patterns**, not a single-motif shortcut.

---

## 3) Reward design & implementation
We implemented a compact, interpretable reward that uses the predictor as the core signal and adds biological constraints:



---

## 4) Reward validation (sanity & smoke tests)
- **Reward sanity test outputs (sample):**
  - `Random:` `0.6966`
  - `TATA inserted:` `0.7830`  → motif increased reward in this sample
  - `High GC:` `-0.6626` → strong negative penalty
  - `Homopolymer:` `-0.2722` → penalized

- **Distribution:**  
  - Min ≈ `-0.688`, Max ≈ `0.993`, Mean ≈ `0.482` — good dynamic range.

- **Hill-climber smoke test (predictor + full reward):**
  - `Initial reward:` `0.4674`
  - Rewards over steps:
    - Step 50: `0.9571`
    - Step 100: `1.0450`
    - Step 150: `1.0691`
    - Step 250: `1.0721`
  - **Final reward:** `1.0793`
  - **Final sequence (example):**
    ```
    GAAAAGGAGGCGGGCGGGGAGAAATGGCGGAGAGTTATACAGTGACGTCATACTTTTACATTAGCAATCCAATAATCCAGAAGGCGTTTGTTTGAACAAAAAGATGTAAGGAATGTAGACTACCAAATAGACCACTAGGGCCCTTGCGGG
    ```

**Interpretation:** simple local search improved the reward quickly and converged to sequences with higher predicted activity while respecting GC / homopolymer constraints - no obvious degeneracy observed.

---

## 5) Findings & takeaways
- The **LightGBM k-6 surrogate** is stable and provides a meaningful ranking signal (Spearman ≈ 0.33 from earlier runs). Percentile calibration makes it numerically robust for RL rewards.
- The **combined reward** (predictor + motif + penalties) has:
  - good dynamic range,
  - is resistant to simple degeneracies (GC collapse, long homopolymers),
  - yields plausible improvement under a hill-climber.
- The surrogate + reward are now **ready to be used** as the fitness function inside an RL environment.

---


## 6) Risks & mitigations
- **Risk:** surrogate artifacts may reflect library-specific biases and agent could exploit them.  
  **Mitigation:** use motif penalties, blacklist, GC/homopolymer penalties; add ensemble or alternate surrogate later; validate designs by biological sanity checks or orthogonal datasets.
- **Risk:** predictor imperfect (Spearman ~0.33) → RL could be noisy.  
  **Mitigation:** percentile mapping + motif terms improve robustness; consider retraining surrogate later with stricter coverage or new features (CNN on GPU).



