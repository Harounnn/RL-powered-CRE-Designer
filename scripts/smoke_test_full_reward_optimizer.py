import random
from src.reward import reward


def random_seq(n=150):
    return "".join(random.choice("ACGT") for _ in range(n))


def mutate(seq):
    i = random.randrange(len(seq))
    bases = "ACGT".replace(seq[i], "")
    return seq[:i] + random.choice(bases) + seq[i + 1:]


def main():
    seq = random_seq()
    r = reward(seq)
    print("Initial reward:", r)

    for step in range(300):
        new_seq = mutate(seq)
        new_r = reward(new_seq)

        if new_r > r:
            seq, r = new_seq, new_r

        if step % 50 == 0:
            print(f"Step {step}: reward={r:.4f}")

    print("\nFinal reward:", r)
    print("Final sequence:", seq)


if __name__ == "__main__":
    main()
