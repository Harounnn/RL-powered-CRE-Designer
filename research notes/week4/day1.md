# Week 4 - Day 1 (W4D1)  RL-Guided Diffusion Environment

## Research framing

The revised research question is:

> Can reinforcement learning adapt a discrete-diffusion generator to produce CRE sequences that better meet an activity objective while preserving sequence quality and diversity learned from SHARPR data?

## Environment design

I wrote docs/env_spec.md to define an MDP over reverse discrete-diffusion trajectories:

- A diffusion prior learns sequence structure from fixed-length SHARPR DNA sequences.
- An episode begins from a highly corrupted sequence and ends with one complete 145-bp DNA candidate.
- The policy chooses reverse-denoising transitions over nucleotide states at each diffusion step. This is not left-to-right autoregressive generation.
- The completed sequence receives a task reward from a replaceable activity-oracle interface.
- RL is regularized toward the frozen pretrained diffusion prior, for example by a KL term, to discourage reward gains that destroy the learned sequence distribution.
- Inpainting can later support modification by holding selected bases fixed while regenerating the rest; it is secondary to de novo generation.

I am keeping the current LightGBM predictor and reward as preliminary scaffolding. They demonstrated a scoring and optimization pipeline, but the modest ranking signal does not make them a final biological objective or permanent model choice.

## Data and scientific scope

The processed dataset notes record 226,538 sequences after DNA coverage >=10 filtering; inspected examples are 145 bp. Earlier smoke tests used random sequences of length 150, so those tests are not length-matched baselines for the planned generator.

The repository currently contains one activity predictor/label path. I can initially study generation for the available assay, but I cannot claim target-versus-off-target cell-type specificity until multi-context data and scoring are available. Predicted reward and biological sequence checks are computational evidence only, not experimental validation.

## Week plan and deliverables

I also wrote docs/week4_plan.md with a revised Day 1–7 schedule:

1. Specify the diffusion MDP (completed today).
2. Define data splits and implement discrete corruption/sampling mechanics.
3. Train and assess a small base diffusion prior.
4. Add the activity-oracle interface, objective conditioning, and optional inpainting.
5. Compare base generation, simple generation/search baselines, and guided sampling under equal oracle-evaluation budgets.
6. Integrate a minimal RL update over diffusion trajectories.
7. Review results and freeze the Week 5 training protocol.

## Literature considered

- [D3 (bioRxiv preprint, 2026 version)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142467/) describes discrete nucleotide-substitution diffusion for regulatory DNA, including activity conditioning and inpainting.
- [DRAKES (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/771e09dd204ea339da0d8114c48afd21-Abstract-Conference.html) frames reward optimization of a pretrained discrete-diffusion model as an RL-style objective with a KL-to-prior term and reports DNA enhancer design experiments.
- [GLID2E (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) is a gradient-free RL-based discrete-diffusion tuning alternative.

These papers inform the design; their results do not establish that this project's data, reward, or implementation will yield functional promoters.

## Current status

I completed Day 1 at the specification level. I have not implemented a diffusion model or RL environment yet, and I did not run tests. The exact corruption schedule, model architecture, denoising horizon, and RL optimizer remain configurable decisions for the planned pilots. A policy-gradient approach can accept the current non-differentiable LightGBM oracle; direct reward backpropagation requires a differentiable oracle.
