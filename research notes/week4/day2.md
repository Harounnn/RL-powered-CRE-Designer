# Week 4 - Day 2 (W4D2) Data Splits and Diffusion Corruption

## Data check

I checked how the SHARPR data was filtered. The script uses total coverage, `rna_total + dna_total >= 10`; it does not filter on DNA coverage alone. I corrected this wording in the environment specification.

The processed table has 242,276 rows. After applying the coverage filter, 226,538 remain. Each retained sequence is a unique 145-base sequence containing only A, C, G, and T, and each has a finite `log2_ratio` value.

## Train, validation, and test splits

I created `scripts/prepare_diffusion_data.py` to prepare the filtered data. It assigns sequences to train, validation, and test partitions in an 80/10/10 split using seed 42. If a sequence appears more than once, all copies go into the same partition. The script saves the three data files, a sequence-to-partition list, and the settings used to make the split in `data/processed/sharpr/diffusion_splits/`.

The split contains:

- Train: 181,230 sequences
- Validation: 22,653 sequences
- Test: 22,655 sequences

I checked that the partitions do not overlap and that every sequence appears in the split list. I ran the script a second time and got the same split.

## Forward corruption

I added `src/diffusion.py` with functions to encode and decode DNA sequences, corrupt sequences, and sample terminal noise. In this pilot, the probability of keeping each original base decreases linearly with the timestep: `alpha_bar(t) = 1 - t/T`, where `T=32`. Bases that are not kept are sampled uniformly from A, C, G, and T. At the final step, the sequence is uniform noise. I chose 32 steps to check the mechanics; the final number of steps will be decided after training and sampling pilots.

## Checks

I checked that step 0 leaves a sequence unchanged, corruption keeps the expected length and DNA alphabet, and the same seed gives the same result. At the final step, the sampled frequencies were close to 25% for each base. I also checked the split sizes, partition overlap, and saved settings.

These checks confirm that the data preparation and corruption code work as intended. They do not show yet that a trained model can generate realistic CRE sequences.

## Next

On Day 3, I will train a small diffusion model using the training split, then use the validation split to check its denoising and generated sequences. I will keep the test split held out. After that, I can decide which model and sampling settings are practical to use in the next stages.
