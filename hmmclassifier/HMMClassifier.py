from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Sequence
import numpy as np
from hmmclassifier.misc import bakis_components, extract_sequence, separate_data
from hmmlearn import hmm


class HMMClassifier:
    """
    Hidden Markov Model classifier using one Gaussian HMM per class.

    The classifier trains a separate :class:`hmmlearn.hmm.GaussianHMM`
    for each label and predicts by selecting the model that yields
    the highest log-likelihood for a sequence.

    Attributes
    ----------
    models : dict
        Mapping from class label to trained :class:`hmmlearn.hmm.GaussianHMM`.
    classes_ : list
        Sorted list of class labels seen during training.
    """

    models: Dict[str, hmm.GaussianHMM]
    classes_: List[str]

    def __init__(self) -> None:
        self.models = {}

    def _fit_separated(
        self,
        separated_data: Dict[str, Tuple[np.ndarray, List[int]]],
        bakis: bool,
        skip: int,
        **kwargs,
    ) -> None:
        """
        Fit one HMM per class using pre-separated training data.

        Parameters
        ----------
        separated_data : dict
            Mapping ``label -> (X, lengths)`` where:

            - ``X`` : ndarray of shape (n_samples, n_features)
            - ``lengths`` : list of sequence lengths

        bakis : bool
            If True, initialize a left-to-right (Bakis) transition topology.

        skip : int
            Number of allowed skip states in the Bakis topology.

        **kwargs
            Additional parameters passed to :class:`hmmlearn.hmm.GaussianHMM`.
        """
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

    def fit_separated(
        self,
        separated_data: Dict[str, Tuple[np.ndarray, List[int]]],
        bakis: bool = False,
        skip: int = 1,
        **kwargs,
    ) -> "HMMClassifier":
        """
        Train the classifier using already separated training data.

        Parameters
        ----------
        separated_data : dict
            Mapping ``label -> (X, lengths)``.

        bakis : bool, default=False
            If True, use a left-to-right Bakis topology.

        skip : int, default=1
            Number of allowed skip transitions between states.

        **kwargs
            Additional arguments passed to
            :class:`hmmlearn.hmm.GaussianHMM`.

        Returns
        -------
        self : HMMClassifier
            Fitted classifier.
        """
        self.classes_ = sorted(separated_data.keys())
        self._fit_separated(separated_data, bakis, skip, **kwargs)
        return self

    def fit(
        self,
        X: np.ndarray,
        lengths: List[int],
        labels: Sequence[str],
        bakis: bool = False,
        skip: int = 1,
        **kwargs,
    ) -> "HMMClassifier":
        """
        Fit the classifier from raw sequential data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Concatenated observation sequences.

        lengths : list of int
            Length of each sequence in ``X``.

        labels : sequence of str
            Class label for each sequence.

        bakis : bool, default=False
            If True, use a Bakis (left-to-right) topology.

        skip : int, default=1
            Number of allowed skip transitions.

        **kwargs
            Additional arguments passed to
            :class:`hmmlearn.hmm.GaussianHMM`.

        Returns
        -------
        self : HMMClassifier
            Fitted classifier.
        """
        separated_data = separate_data(X, lengths, labels)
        self.fit_separated(separated_data, bakis, skip, **kwargs)
        return self

    def decision_function(
        self,
        X: np.ndarray,
        lengths: List[int] | None = None,
    ) -> np.ndarray:
        """
        Compute log-likelihood scores for each class.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Observation sequences.

        lengths : list of int, optional
            Sequence lengths. If None, ``X`` is treated as a single sequence.

        Returns
        -------
        scores : ndarray of shape (n_sequences, n_classes)
            Log-likelihood scores for each sequence under each model.
        """
        if lengths is None:
            lengths = [len(X)]

        sequences = extract_sequence(X, lengths)
        scores = np.zeros((len(sequences), len(self.classes_)))

        for i, seq in enumerate(sequences):
            for j, label in enumerate(self.classes_):
                scores[i, j] = self.models[label].score(seq)

        return scores

    def predict(
        self,
        X: np.ndarray,
        lengths: List[int] | None = None,
    ) -> List[str]:
        """
        Predict class labels for sequences.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Observation sequences.

        lengths : list of int, optional
            Sequence lengths. If None, ``X`` is treated as a single sequence.

        Returns
        -------
        labels : list of str
            Predicted label for each sequence.
        """
        scores = self.decision_function(X, lengths)
        indices = scores.argmax(axis=1)
        return [self.classes_[i] for i in indices]

    def save(self, path: str | Path) -> None:
        """
        Save the classifier to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            File path where the model will be saved.
        """
        with Path(path).open("wb") as f:
            pickle.dump((self.classes_, self.models), f)

    @classmethod
    def load(cls, path: str | Path) -> "HMMClassifier":
        """
        Load a classifier from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            File path containing a saved classifier.

        Returns
        -------
        model : HMMClassifier
            Loaded classifier instance.
        """
        self = cls()
        with Path(path).open("rb") as f:
            self.classes_, self.models = pickle.load(f)
        return self