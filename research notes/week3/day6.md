# Week 3 — Day 6 (W3D6) — Predictor and Reward Follow-up

## What we did

Day 6 was a Week 3 buffer day, so we reran the existing predictor, reward, and optimizer checks to consolidate the Week 3 results before starting the RL environment work.

We ran these existing scripts from the project root as Python modules:

- `scripts/test_predictor_wrapper.py` — checked predictor scores for random sequences, inserted a TATA motif in one sample, and checked that scores vary across sequences.
- `scripts/test_full_reward.py` — checked the combined reward on random, TATA-inserted, high-GC, and homopolymer sequences, then sampled its distribution across 1,000 random sequences.
- `scripts/smoke_test_full_reward_optimizer.py` — ran a 300-step hill-climber that proposes one-base substitutions and keeps mutations that increase the full reward.

## Results

### Predictor wrapper

- A random sequence scored **0.569** on the percentile scale.
- Across 1,000 random sequences, scores ranged from **0.019 to 0.983**, with a mean of **0.579**.
- In the sample motif check, inserting `TATAAA` raised the score from **0.794 to 0.865**.
- The ordering check returned varied scores, from about **0.256 to 0.894** for its ten sampled sequences.

This confirms that the wrapper runs and assigns a broad range of percentile scores to random sequences. The single TATA insertion result is only a sanity check; it does not show that TATA always raises the predictor score.

### Full reward

- Random sequence: **0.560**
- Same sample with TATA inserted: **0.666**
- High-GC sequence (`G` repeated 150 times): **−0.663**
- Homopolymer sequence (`A` repeated 150 times): **−0.272**
- Across 1,000 random sequences: minimum **−0.620**, maximum **0.945**, mean **0.502**

The stress cases received negative rewards, as intended by the GC and homopolymer penalties. The sampled distribution shows that the combined reward has room to distinguish sequences, but does not demonstrate that its weights are optimal or that high-reward sequences are biologically functional.

### Hill-climber smoke test

The optimizer started at reward **0.124** and ended at **1.034** after 300 proposed single-base substitutions. The accepted sequence mutations improved the objective measured by the current predictor-plus-constraints reward.

This demonstrates that simple local search can increase the computational reward. It does not validate promoter activity, and improvement may reflect properties or biases of the surrogate and reward design.

## Week 3 takeaway

The follow-up runs confirm that the predictor wrapper, combined reward, and simple hill-climber work together on the available artifacts. The surrogate provides a varied ranking signal, the reward penalizes the tested extreme sequences, and local search can find sequences with higher reward. These are software and optimization sanity checks, not experimental evidence of promoter function.

## What's next — Week 4

Move from standalone reward optimization to a defined RL environment:

1. Specify the environment state, actions, transitions, reward, constraints, and episode termination in `docs/env_spec.md`.
2. Implement a small sequence-edit action space, starting with single-base substitutions if that remains appropriate.
3. Connect the environment to `CREPredictor` and the full reward.
4. Check environment behavior and reproducibility.
5. Compare random search and hill-climbing under the same evaluation budget.
6. Choose and add a minimal RL training loop after the environment and baselines are established.

Keep predicted reward separate from evidence of real promoter activity throughout the next phase.
