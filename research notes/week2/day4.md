## Week 2 – Day 4: Why We Ran Encoding Experiments

**Experiment**
- We tested one-hot, overlapping k-mer, and k-mer count encodings to understand how each representation captures sequence features—especially motifs like the TATA box—versus simpler base-level structure.
- This helps us evaluate which encoding gives the strongest signal for detecting promoter patterns in small, interpretable examples.

**Results**
- **One-hot encoding** preserves exact nucleotide order and position but doesn’t provide contextual similarity or feature counting.
- **Overlapping k-mer counts** highlight repeated motifs (TATA) that one-hot alone might obscure, making it more informative for motif-driven tasks.
- **k-mer counts produce structured feature vectors** easy to feed into simple classifiers or later embed into models.

**Upcoming work:**
- These insights will guide our choice of input format for building the CRE predictor.
- For instance, we may opt to start with k-mer-based features (good context and compact representation) before moving to richer models like CNNs over one-hot or even pretrained embeddings.
