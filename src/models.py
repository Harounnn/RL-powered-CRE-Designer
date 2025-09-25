"""Baseline model wrappers (sklearn)"""
from sklearn.linear_model import LogisticRegression
import joblib


class KmerLogReg:
    def __init__(self, C=1.0):
        self.clf = LogisticRegression(max_iter=1000, solver='liblinear', C=C)

def fit(self, X, y):
    self.clf.fit(X, y)
    return self


def predict(self, X):
    return self.clf.predict(X)


def predict_proba(self, X):
    return self.clf.predict_proba(X)[:,1]


def save(self, path):
    joblib.dump(self.clf, path)


def load(self, path):
    self.clf = joblib.load(path)
    return self