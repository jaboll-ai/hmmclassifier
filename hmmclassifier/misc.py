from collections import defaultdict
from typing import Dict, List, Tuple, Sequence
import numpy as np

def separate_data(
    X: np.ndarray,
    lengths: Sequence[int],
    labels: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    """
    Separate concatenated sequences by class label.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Concatenated observation sequences.

    lengths : sequence of int
        Length of each sequence contained in ``X``.

    labels : sequence of str
        Class label for each sequence.

    Returns
    -------
    separated : dict
        Mapping ``label -> (X_label, lengths_label)`` where:

        - ``X_label`` : ndarray of shape (n_samples_label, n_features)
          Concatenated observations belonging to the label.
        - ``lengths_label`` : list of int
          Lengths of individual sequences for that label.
    """
    assert len(lengths) == len(labels), "Invalid dimensions"

    sequences = extract_sequence(X, lengths)
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)

    for seq, label in zip(sequences, labels):
        grouped[label].append(seq)

    separated: Dict[str, Tuple[np.ndarray, List[int]]] = {}

    for label, seqs in grouped.items():
        label_lengths = [len(s) for s in seqs]
        label_X = np.concatenate(seqs, axis=0)

        separated[label] = (label_X, label_lengths)

    return separated


def extract_sequence(
    X: np.ndarray,
    lengths: Sequence[int],
) -> List[np.ndarray]:
    """
    Extract individual sequences from concatenated observations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Concatenated observation sequences.

    lengths : sequence of int
        Length of each sequence contained in ``X``.

    Returns
    -------
    sequences : list of ndarray
        List of arrays representing individual sequences.
    """
    assert sum(lengths) == len(X), "Invalid dimensions"

    sequences: List[np.ndarray] = []
    start = 0

    for l in lengths:
        end = start + l
        sequences.append(X[start:end])
        start = end

    return sequences


def bakis_components(
    n_components: int,
    skip: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate transition and start probabilities for a Bakis HMM topology.

    A Bakis (left-to-right) HMM restricts transitions so that states
    can only remain the same or move forward.

    Parameters
    ----------
    n_components : int
        Number of hidden states.

    skip : int
        Maximum number of states that may be skipped in a transition.

        For example:

        - ``skip = 0`` → only self-loop and next state
        - ``skip = 1`` → allow transitions up to two states ahead

    Returns
    -------
    transmat : ndarray of shape (n_components, n_components)
        Transition probability matrix.

    startprob : ndarray of shape (n_components,)
        Initial state probability distribution.
    """
    transmat = np.zeros((n_components, n_components))

    for i in range(n_components):
        max_jump = min(skip + 1, n_components - i - 1)
        targets = [i + j for j in range(max_jump + 1)]
        prob = 1.0 / len(targets)

        for t in targets:
            transmat[i, t] = prob

    startprob = np.zeros(n_components)
    startprob[0] = 1.0

    return transmat, startprob