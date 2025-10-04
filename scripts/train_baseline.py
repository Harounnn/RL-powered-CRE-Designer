"""Train baseline logistic regression on k-mer features."""
import json
import numpy as np
from src.features import build_feature_matrix
from src.models import KmerLogReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# load synthetic data
with open('data/synthetic_pos.json') as f:
    pos = json.load(f)
with open('data/synthetic_neg.json') as f:
    neg = json.load(f)


seqs = pos + neg
labels = [1]*len(pos) + [0]*len(neg)


X, vocab = build_feature_matrix(seqs, k=6)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)


model = KmerLogReg()
model.fit(X_train, y_train)


pred = model.predict(X_test)
prob = model.predict_proba(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
print('AUC:', roc_auc_score(y_test, prob))


model.save('models/kmer_logreg.joblib')