## Notes

### Experiment
We used logistic regression as a baseline promoter predictor and generated a small synthetic dataset.  
The goal was to test whether **k-mer encoding** (overlapping k-mer count vectors) provides a useful, simple representation for detecting promoter-like motifs.

### Results
K-mer features work well in this controlled setting: sequences containing a TATA-like motif are clearly separable from negatives, and the classifier assigns high weight to TATA-related k-mers. We also verified this behavior by evaluating the model on whole DNA sequences with and without the TATA box.

### Caveats & future work
The perfect accuracy (1.0) likely reflects the synthetic, low-noise setup (fixed motif insertion and balanced classes). The next steps are to probe robustness and overfitting before committing to a production predictor:
- Test with motifs placed at random positions and with noisy / mutated motifs.  
- Vary k (e.g., 4, 5, 6) and compare separability.  
- Introduce realistic negative examples (GC-rich sequences, random genomic background) and class imbalance.  
- Move toward Week 3: prepare a small real MPRA/promoter dataset and evaluate whether k-mer features still perform; if not, try one-hot + CNN or hybrid tokenization.

Our aim now is not to optimize the baseline accuracy but to choose an encoding that generalizes well to more realistic data.
