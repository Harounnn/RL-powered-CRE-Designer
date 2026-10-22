# RL Environment Specification : Discrete-Diffusion CRE Design

**Status:** Day 1 design specification. This defines the prototype interface; it does not claim that an environment or diffusion model has been implemented.

## 1. Purpose and scope

The environment trains an RL policy to guide the reverse-denoising process of a discrete-diffusion model that generates synthetic cis-regulatory element (CRE) sequences. The primary task is de novo generation. A secondary task may use inpainting to preserve selected bases from a supplied sequence while regenerating the remaining positions.

The central experiment is whether RL increases objective reward over the unoptimized diffusion prior while retaining sequence quality and diversity. I treat the activity predictor and reward currently in the repository as preliminary components and will keep them replaceable.

## 2. Available data and initial scope

- Primary training material: processed SHARPR sequences, with 226,538 rows after filtering on total read coverage (`rna_total + dna_total >= 10`). The filter was confirmed in `experiments/make_k6_and_baseline.py`; it is not a DNA-only threshold.
- The inspected processed examples are 145 bases long. The primary environment therefore fixes sequence length at 145; reject other lengths unless a separately documented task changes this contract.
- Sequence alphabet: `A`, `C`, `G`, `T`. Ambiguous symbols are rejected by the environment.
- The current repository has one activity-label/predictor path. The first objective is therefore conditioned on the available assay. The interface may carry an assay/target identifier, but cell-type-specific or target-versus-off-target claims require multi-context data and predictors that are not currently present.
- Establish explicit unique-sequence train/validation/test partitions before training the generative model. Day 2 uses 80/10/10 with seed 42. Do not use validation or test sequences in generator training or RL optimization. Keep duplicate sequences in one partition.

## 3. Generative model and MDP

Use a categorical discrete-diffusion model over fixed-length nucleotide sequences. A forward corruption process progressively substitutes nucleotides until the terminal noisy sequence is uniform. The learned reverse process maps a noisy sequence toward a data-like sequence. The Day 2 mechanics pilot uses a D3-inspired uniform-substitution kernel with linear cumulative retention, `alpha_bar(t) = 1 - t/T`, and `T=32`; this is a configurable mechanics setting, not a final training horizon or model choice. Architecture and training settings remain for the Day 3 pilot.

### State / observation

At denoising step `t`, the observation contains:

- Current noisy sequence `x_t` of length 145, encoded as categorical tokens or a 145Ã—4 one-hot tensor.
- Normalized timestep `t/T` (and any schedule value needed by the policy).
- Design condition `c`: initially the available assay plus the objective `maximize_activity`.
- For inpainting mode only: a fixed-position mask and the original bases at those positions.

The full sequence and timestep are included because the reward depends on sequence context and because the transition policy is time-dependent. No separate edit history is needed for a Markov reverse process.

### Initial state

For de novo generation, sample `x_T` from the terminal distribution of the forward process, approximately uniform over A/C/G/T at each position. Seed all random generators and record the seed. In inpainting mode, initialize unmasked positions from noise and preserve the specified bases according to the diffusion inpainting procedure.

### Action

At each timestep, the policy selects a reverse-denoising transition to `x_{t-1}`. The transition is represented by categorical nucleotide choices at positions according to a factorized or otherwise tractable learned reverse kernel; it is not a left-to-right next-base action. RL updates the reverse policy, initialized from a pretrained sequence prior.

For a score-function policy-gradient implementation, store the transition log-probability (or a tractable per-position factorization) for the sampled action. The environment interface should not require reward gradients.

### Transition

Apply the sampled reverse transition and decrement `t`. In inpainting mode, clamp/restore masked positions according to the defined diffusion procedure. The transition must preserve sequence shape and alphabet. The episode ends when `t=0` and a complete 145-bp candidate `x_0` is produced.

## 4. Objective and reward

Let `R_task(x_0, c)` be the completed-sequence task reward and `P_0` the frozen pretrained diffusion policy. The RL objective is conceptually:

`maximize E[R_task(x_0, c)] - beta * KL(P_RL(trajectory | c) || P_0(trajectory | c))`

where `beta` controls how strongly the RL policy must remain near the reference generator. The KL term is a prior-preservation regularizer, not evidence of biological function. Its practical form (per-step transition KL or a tractable sequence-level approximation) is an implementation decision to validate in the Day 6 pilot.

### Preliminary task reward

Initially, `R_task` may call the existing predictor/reward behind a replaceable scoring interface. Its present ingredients are predictor percentile, simple motif bonus, GC deviation penalty, homopolymer penalty, and blacklist penalty. Treat weights and motifs as provisional. The current LightGBM oracle is non-differentiable; a policy-gradient RL method can use it, while direct reward backpropagation methods require a differentiable predictor/reward path.

The first objective is maximizing activity percentile for the available assay. Requesting arbitrary target activity levels in native units requires a calibrated predictor that supports those targets. Do not use the current single-assay signal to imply cell-type specificity.

Reward is evaluated on the completed sequence. Do not apply the length-calibrated percentile score to partial sequences. Record component scores separately for analysis. A later implementation may add valid shaping, but it must preserve the intended terminal objective and be described explicitly.

## 5. Episode mechanics and constraints

- One episode is one reverse-diffusion trajectory from `x_T` to a complete `x_0`.
- The horizon is the configured number of reverse steps `T`; choose it after a short compute/sampling pilot rather than inventing a fixed value here.
- The de novo task has no initial promoter seed. Optional inpainting is a separate mode and must be reported separately.
- Hard validity rules: sequence length 145 and alphabet `ACGT`; invalid transitions are implementation errors, not soft reward penalties.
- GC, homopolymer, blacklist, novelty, and sequence-distribution objectives are reported/soft-scored unless a later experiment explicitly defines a hard biological constraint.
- The pretrained sequence model is retained as a frozen reference during RL updates.

## 6. Randomness and reproducibility

Record and seed Python, NumPy, and PyTorch RNGs; save data split identifiers, training and sampler configs, diffusion checkpoint, reward-oracle version, policy checkpoint, condition, sampler settings, and generated sequences. Save per-trajectory denoising states or sufficient log-probability summaries for RL updates. Deterministic GPU execution is not assumed; document hardware and reproducibility limits.

## 7. Evaluation and baselines

Count cost in complete-sequence oracle evaluations and compare methods under a common evaluation budget. Baselines should include:

1. Frozen pretrained discrete-diffusion prior with no RL reward optimization.
2. Random and/or simple empirical/Markov sequence generation.
3. Existing random-start hill-climber as a modifier baseline, clearly labeled as a different task.
4. RL-updated discrete-diffusion policy.
5. Inference-time reward guidance, if practical, to distinguish RL fine-tuning from sampling-time steering.

Report separately:

- Task-oracle reward and activity score distributions.
- Sequence validity, uniqueness, pairwise diversity, and nearest-neighbor similarity to the generator training set.
- GC distribution, homopolymer/blacklist rates, and motif summaries.
- KL/reference-model likelihood or another predeclared prior-preservation measure.
- Performance under an independent held-out or cross-fitted activity model where feasible. This is a robustness check, not experimental validation.

## 8. Modification extension

The same diffusion model can support sequence modification through inpainting: hold selected bases fixed and denoise the remaining positions. This is secondary to de novo generation and should only be added after unconditional generation and the RL trajectory interface work.

## 9. Key risks

- The POC LightGBM model has only modest reported rank performance (Spearman about 0.33) and may be exploitable outside its training distribution.
- Reward maximization may reduce diversity or drive the policy away from the learned sequence prior; monitor task reward and prior-preservation metrics together.
- Current data do not provide an explicit multi-cell-type objective in the repo. Cell-type-aware generation requires matched contexts and target/off-target scoring.
- Model-generated or high-scoring sequences are computational candidates, not experimentally validated promoters.

## 10. Literature basis

- [D3: Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion (bioRxiv preprint, 2026 version)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142467/) motivates substitution-based categorical diffusion, conditional activity generation, and inpainting for CREs.
- [DRAKES, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/771e09dd204ea339da0d8114c48afd21-Abstract-Conference.html) provides a directly relevant RL-style reward optimization formulation for pretrained discrete diffusion models, with a KL-to-prior term and DNA enhancer design experiments.
- [GLID2E, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) is a gradient-free RL-based alternative worth considering if a lightweight policy update is needed.
