## Notes

**Summary**

- **One-hot** is a *base-level encoding*, not an embedding. It maps each nucleotide (A/C/G/T) to a binary vector so models can ingest sequences numerically; any *embedding* is learned later by the model.  
- **k-mers** (overlapping substrings like 6-mers) add **local context** beyond single bases and are widely used in genomic LMs (e.g., DNABERT uses overlapping k-mers).
- **BPE/subword tokenization** builds a **data-driven vocabulary** of frequent substrings, enabling **variable-length tokens** and often better modeling of longer patterns (e.g., DNABERT-2 uses BPE). 

**Nuances from recent work**

- Overlapping k-mers give context, but in masked-LM pretraining they can **leak information** because adjacent tokens share k−1 bases. 
- BPE can learn longer units but may **amplify base-frequency biases**. 
- Tokenization choice measurably affects downstream tasks; comparative studies analyze alternatives across biosequence problems.
- **Hybrid strategies (k-mer + BPE)** are emerging to capture both local motifs and broader context (e.g., 6-mer + BPE-600). Early reports suggest benefits over using either alone. 

**Takeaways**

- For a first surrogate predictor, **one-hot** or **fixed k-mer one-hot** is simple and stable.  
- To capture motif+context for promoters, consider **overlapping 6-mers** or **BPE**; keep in mind the k-mer leakage issue and BPE bias.
- Later, we can experiment with a **hybrid tokenizer** to see if it improves promoter-strength prediction. 
