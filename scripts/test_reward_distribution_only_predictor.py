import random
from src.predictor import CREPredictor


def random_seq(n=150):
    return "".join(random.choice("ACGT") for _ in range(n))


def main():
    predictor = CREPredictor()

    print("\n=== Reward distribution (predictor only) ===")
    rewards = [predictor.percentile_score(random_seq()) for _ in range(2000)]

    print("Min:", min(rewards))
    print("Max:", max(rewards))
    print("Mean:", sum(rewards) / len(rewards))

    # Rough histogram
    bins = [0] * 10
    for r in rewards:
        bins[int(r * 10) if r < 1 else 9] += 1

    print("\nHistogram (10 bins):")
    for i, c in enumerate(bins):
        print(f"{i/10:.1f}-{(i+1)/10:.1f}: {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
