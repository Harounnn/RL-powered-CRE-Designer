## 1) Raw inspector outputs (copied from your run)

```
{'seq': 'ACGTCTATAAAGGCTGACCT', 'predictor': 0.6, 'pwm_best': np.float64(1.0), 'pwm_idx': 5, 'gc_score': 0.8824969025845955, 'homopolymer_score': 1.0, 'positional_bonus': 1.0, 'blacklist_ok': True, 'total': np.float64(2.0529987610338383)}

{'seq': 'ACGTCTACAAAGGCTGACCT', 'predictor': 0.6, 'pwm_best': np.float64(0.14285714285714288), 'pwm_idx': 3, 'gc_score': 1.0, 'homopolymer_score': 1.0, 'positional_bonus': 0.0, 'blacklist_ok': True, 'total': np.float64(1.385714285714286)}

{'seq': 'GCGCGCGCGCGCGCGCGCGC', 'predictor': 0.0, 'pwm_best': np.float64(0.00041649312786339054), 'pwm_idx': 0, 'gc_score': 3.7266531720786777e-06, 'homopolymer_score': 1.0, 'positional_bonus': 0.0, 'blacklist_ok': True, 'total': np.float64(0.30025138653798683)}

{'seq': 'ACGTCTATCATATATGACCT', 'predictor': 0.6, 'pwm_best': np.float64(1.0), 'pwm_idx': 10, 'gc_score': 0.32465246735834974, 'homopolymer_score': 1.0, 'positional_bonus': 0.0, 'blacklist_ok': True, 'total': np.float64(1.6298609869433398)}

{'seq': 'GAATTCATCATATATGACCT', 'predictor': 0.7, 'pwm_best': np.float64(1.0), 'pwm_idx': 10, 'gc_score': 0.1353352832366127, 'homopolymer_score': 1.0, 'positional_bonus': 0.0, 'blacklist_ok': False, 'total': np.float64(0.0)}
```


---

## 2) Quick interpretation (what each column tells us)

- **predictor**: surrogate model score (toy predictor). Higher → sequence predicted to be stronger promoter.  
- **pwm_best**: normalized PWM match (0..1). `1.0` means a perfect motif match was found somewhere in the sequence.  
- **pwm_idx**: index where the best PWM match was found (0-based).  
- **gc_score**: smooth score in [0..1] that is high when GC% ≈ 50% (sigma used in your code).  
- **homopolymer_score**: 1.0 means no problematic long runs; lower → penalty for long homopolymers.  
- **positional_bonus**: 1.0 if best motif sits inside the target index window (here `(5,9)`), else 0.0.  
- **blacklist_ok**: False if a disallowed motif (e.g., `GAATTC`) was detected. In your current code a blacklist hit zeros the total.  
- **total**: weighted sum of components (subject to blacklist multiplier).

---

## 3) Remarks & suggestions (practical tuning / next experiments)

1. **Blacklist handling is too harsh as a multiplier.**  
   - *Effect observed:* seq_E has good sub-scores but total → 0.0.  
   - *Suggestion:* convert blacklist hard-zero into a *penalty* (e.g., subtract fixed value or multiply by 0.2) so these sequences are penalized but not indistinguishable from absolute failure. Example: `total = max(0, total - blacklist_penalty)` or `total *= 0.2`.

2. **Positional window sensitivity.**  
   - seq_D had motif at index 10 (you expected it to matter), but positional bonus window `(5,9)` excluded it.  
   - *Suggestion:* either widen the target_range or make the positional bonus a smooth function of distance (e.g., Gaussian centered at preferred index).

3. **GC scoring behavior depends strongly on `sigma`.**  
   - seq_C had near-0 gc_score because sigma was small (0.1). If you want less punishing GC tolerance, increase sigma to widen acceptable GC range.  
   - *Suggestion:* try `sigma=0.15` or `0.2` and see effect on gc_score.

4. **Predictor realism matters.**  
   - Right now a toy predictor drives α-term. Replacing it with a trained CNN (Week 3) will substantially change rankings — keep logging components separately so you can see how the real predictor shifts priorities.

5. **Weight tuning will change sequence ranking.**  
   - The current totals reflect your chosen weights. If motif presence is too dominant/weak, adjust `beta`. If GC should matter less, reduce `gamma` (or vice-versa).

6. **Add more diagnostics**  
   - Run a small grid of sequences that systematically vary motif position / GC / blacklist presence to visualize how `total` changes across axes. Plotting will reveal sensitivity.
