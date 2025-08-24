## Notes

One-hot encoding represents each nucleotide (`A`, `C`, `G`, `T`) as a unique vector of numbers, making sequences directly usable by mathematical models.  

This is different from **k-mers** or **BPE (byte-pair encoding)**:
- **One-hot** → focuses on direct numeric representation of each base.  
- **k-mers / BPE** → are tokenization strategies that break the sequence into meaningful subunits. These tokens are later embedded (possibly also using one-hot, or more advanced embeddings).  

Thus, one-hot is a *base-level representation*, while k-mers and BPE are *higher-level chunking methods*. Both can be combined in practice.
