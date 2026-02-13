import random
from src.predictor import CREPredictor


def random_seq(n=150):
    return "".join(random.choice("ACGT") for _ in range(n))


def main():
    predictor = CREPredictor()

    print("\n=== Basic sanity ===")
    seq = random_seq()
    print("Raw:", predictor.raw_score(seq))
    print("Percentile:", predictor.percentile_score(seq))

    print("\n=== Random distribution ===")
    scores = [predictor.percentile_score(random_seq()) for _ in range(1000)]
    print("Min:", min(scores), "Max:", max(scores), "Mean:", sum(scores)/len(scores))

    print("\n=== Motif enrichment test ===")
    base = random_seq()
    tata = base[:70] + "TATAAA" + base[76:]
    print("Random seq score:", predictor.percentile_score(base))
    print("TATA inserted score:", predictor.percentile_score(tata))

    print("\n=== Ordering test ===")
    seqs = [random_seq() for _ in range(10)]
    scores = [(s, predictor.percentile_score(s)) for s in seqs]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    for s, sc in scores_sorted:
        print(sc)

    print("\nAll tests done.")


if __name__ == "__main__":
    main()
