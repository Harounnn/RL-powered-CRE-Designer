import json
import numpy as np
import lightgbm as lgb


class CREPredictor:
    def __init__(
        self,
        model_path="models/predictor_sharpr_lightgbm.txt",
        vocab_path="data/processed/sharpr/vocab_k6.json",
        percentiles_path="models/predictor_percentiles.npy",
        k=6,
    ):
        self.model = lgb.Booster(model_file=model_path)

        with open(vocab_path, "r") as f:
            vocab_raw = json.load(f)

        # Handle vocab stored as list or dict
        if isinstance(vocab_raw, list):
            self.vocab = {kmer: i for i, kmer in enumerate(vocab_raw)}
        elif isinstance(vocab_raw, dict):
            # If dict is token -> index already
            if isinstance(next(iter(vocab_raw.values())), int):
                self.vocab = vocab_raw
            else:
                # reverse index -> token
                self.vocab = {v: int(k) for k, v in vocab_raw.items()}
        else:
            raise ValueError("Unsupported vocab format")

        self.k = k
        self.sorted_preds = np.load(percentiles_path)
        self.n = len(self.sorted_preds)

    def _to_features(self, seq: str):
        seq = seq.upper().replace("N", "A")
        counts = np.zeros(len(self.vocab), dtype=np.float32)

        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k]
            idx = self.vocab.get(kmer)
            if idx is not None:
                counts[idx] += 1.0

        return counts.reshape(1, -1)

    def raw_score(self, seq: str) -> float:
        X = self._to_features(seq)
        return float(self.model.predict(X)[0])

    def percentile_score(self, seq: str) -> float:
        raw = self.raw_score(seq)
        rank = np.searchsorted(self.sorted_preds, raw, side="right")
        return rank / self.n


if __name__ == "__main__":
    predictor = CREPredictor()
    test_seq = "ACGT" * 50
    print("Raw:", predictor.raw_score(test_seq))
    print("Percentile:", predictor.percentile_score(test_seq))
