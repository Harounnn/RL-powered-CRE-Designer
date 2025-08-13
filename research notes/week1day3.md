## Understanding the reward function for CRE design agent

### Key Formula
$$
\text{reward}(seq) = \alpha \cdot \text{predictor\_score}(seq) 
+ \beta \cdot \text{best\_PWM\_score}(seq, motif) 
- \gamma \cdot \text{gc\_penalty}(seq) 
- \delta \cdot \text{homopolymer\_penalty}(seq)
$$

---

## Components

1. **Predictor score (α)**  
   - Output of an ML model predicting promoter strength in a target cell type.  
   - Higher score → more biologically effective CRE.

2. **Best PWM score (β)**  
   - Measures the strongest match of a desired motif in the sequence.  
   - Encourages the agent to place key motifs like the TATA box correctly.

3. **GC penalty (γ)**  
   - Penalizes deviation from optimal GC% (e.g., 40–60%).  
   - Too high/low GC% = less stability and biological realism.

4. **Homopolymer penalty (δ)**  
   - Penalizes long repeats of the same base (e.g., “AAAAAA”).  
   - Prevents synthesis/fidelity issues.
