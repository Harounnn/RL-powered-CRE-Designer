# Week 4 Plan — Discrete-Diffusion Generation and RL

## Research question

Can reinforcement learning adapt a discrete-diffusion generator to produce CRE sequences that better meet an activity objective while preserving sequence quality and diversity learned from SHARPR data?

The project studies RL over reverse-denoising trajectories. It is not autoregressive fine-tuning, a generator-only study, or local-mutation optimization. Activity scores are computational surrogate results, not experimental validation. The available repository currently has one activity predictor/label path, so this week does not establish multi-cell-type specificity.

## Day-by-day plan

1. **Day 1 — Specify the environment (complete).** Define the de novo generation MDP, sequence contract, terminal reward, frozen-prior regularization, scope limits, and evaluation axes in `docs/env_spec.md`.
2. **Day 2 — Prepare data and corruption mechanics (complete).** Audit processed SHARPR data; filter on `rna_total + dna_total >= 10`; create deterministic, sequence-level 80/10/10 train/validation/test partitions; implement categorical uniform-substitution corruption and terminal-noise sampling. The 32-step linear-retention schedule is a mechanics pilot only.
3. **Day 3 — Train a small diffusion prior.** Fit a compact denoiser on training sequences, assess held-out denoising and unconditional samples, and choose a practical horizon/configuration from measured behavior and compute.
4. **Day 4 — Add objective scoring and conditioning.** Put the activity predictor behind a replaceable scorer interface; define the first available-assay condition and terminal objective. Consider inpainting only if unconditional generation and scoring integration are stable.
5. **Day 5 — Establish comparable baselines.** Compare frozen-prior generation with simple empirical generation and the existing search baseline under a matched complete-sequence oracle-evaluation budget. Report reward, validity, diversity, and distribution checks separately.
6. **Day 6 — Add a minimal RL update.** Optimize reverse-diffusion trajectories with a score-function method compatible with the non-differentiable LightGBM prototype; include a KL-to-frozen-prior term and run a short pipeline smoke run.
7. **Day 7 — Review and set the next protocol.** Summarize results and limitations, identify failure modes, and select the Week 5 training/evaluation protocol.

## Working decisions

- The primary task is de novo generation of fixed-length 145-base sequences over A/C/G/T; modification through inpainting is secondary.
- Split by unique sequence before generative model training; keep duplicate sequences together and reserve validation/test data from generator and RL optimization.
- LightGBM and the current reward are replaceable scaffolding. Do not interpret predicted activity as measured promoter function.
- Track activity objective, sequence validity, diversity, distribution fidelity, and biological checks separately. Monitor prior preservation during RL.
- Record configuration, random seeds, data split identifiers, scorer version, and generated sequences for reproducibility.
