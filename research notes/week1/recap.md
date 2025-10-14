# Week 1 Summary

## Key Learnings

- **Cis-Regulatory Elements (CREs)** are short, non-coding DNA regions (like promoters and enhancers) that regulate the transcription of nearby genes by binding transcription factors.
- **Promoters** help recruit the transcriptional machinery (e.g. RNA polymerase II), often featuring motifs such as the **TATA box** around 25–35 bp upstream of the transcription start site.
- **Motifs** are short, recurring DNA patterns with functional roles (such as indicating the transcription start region) and can be modeled using Position Weight Matrices (PWMs), which quantify the likelihood of each nucleotide at each motif position.
- **PWM scoring** assigns a match score to any DNA window by estimating how well it aligns with the motif pattern—this score can be normalized for use in model-based reward signals.
- We defined an **RL-ready reward function** integrating several biological criteria:
  - **Predictor score** – simulates promoter strength prediction.
  - **PWM match score** – encourages motif presence.
  - **GC content penalty** – maintains realistic base composition.
  - **Homopolymer penalty** – discourages problematic repeats.
  - **Positional bonus** and **blacklist filtering** – control motif positioning and exclude unwanted sequences.

This week built a solid bridge between core molecular biology and the reward mechanics that will steer our future RL agent.
