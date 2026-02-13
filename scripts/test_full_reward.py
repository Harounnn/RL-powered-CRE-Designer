import random
from src.reward import reward


def random_seq(n=150):
    return "".join(random.choice("ACGT") for _ in range(n))


def main():
    print("\n=== Reward sanity tests ===")

    base = random_seq()
    tata = base[:70] + "TATAAA" + base[76:]
    high_gc = "G" * 150
    homopolymer = "A" * 150

    print("Random:", reward(base))
    print("TATA inserted:", reward(tata))
    print("High GC:", reward(high_gc))
    print("Homopolymer:", reward(homopolymer))

    print("\n=== Distribution ===")
    rewards = [reward(random_seq()) for _ in range(1000)]
    print("Min:", min(rewards))
    print("Max:", max(rewards))
    print("Mean:", sum(rewards)/len(rewards))


if __name__ == "__main__":
    main()
