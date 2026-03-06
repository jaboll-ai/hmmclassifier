from pathlib import Path
import pickle
from hmmclassifier.misc import bakis_components, extract_sequence, separate_data
from hmmlearn import hmm
import numpy as np

class HMMClassifier():
    def __init__(self):
        self.models = {}

    def _fit_separated(self, separated_data: dict, bakis: bool, skip: int, **kwargs):
        self.models = {}
        params = dict(kwargs)
        params.setdefault("n_components", 3)
        params.setdefault("covariance_type", "diag")
        params.setdefault("n_iter", 200)
        params.setdefault("random_state", 42)
        if bakis:
            params["init_params"] = "mc"
            params["params"] = "mc"
        for label in self.classes_:
            X_l, lengths_l = separated_data[label]
            model = hmm.GaussianHMM(**params)
            if bakis:
                transmat, startprob = bakis_components(params["n_components"], skip)
                model.startprob_ = startprob
                model.transmat_ = transmat
            model.fit(X_l, lengths_l)
            self.models[label] = model

    def fit_separated(self, separated_data: dict, bakis: bool = False, skip: int = 1, **kwargs):
        self.classes_ = sorted(separated_data.keys())
        self._fit_separated(separated_data, bakis, skip, **kwargs)
        return self

    def fit(self, X, lengths, labels, bakis: bool = False, skip: int = 1, **kwargs):
        separated_data = separate_data(X, lengths, labels)
        self.fit_separated(separated_data, bakis, skip, **kwargs)
        return self

    def decision_function(self, X, lengths=None):
        if lengths is None: lengths = [len(X)]
        sequences = extract_sequence(X, lengths)
        scores = np.zeros((len(sequences), len(self.classes_)))
        for i, seq in enumerate(sequences):
            for j, label in enumerate(self.classes_):
                scores[i, j] = self.models[label].score(seq)
        return scores

    def predict(self, X, lengths=None):
        scores = self.decision_function(X, lengths)
        indices = scores.argmax(axis=1)
        return [self.classes_[i] for i in indices]

    def save(self, path: str|Path):
        with Path(path).open("wb") as f:
            pickle.dump((self.classes_, self.models), f)

    @classmethod
    def load(cls, path: str|Path):
        self = cls()
        with Path(path).open("rb") as f:
            self.classes_, self.models = pickle.load(f)
        return self
