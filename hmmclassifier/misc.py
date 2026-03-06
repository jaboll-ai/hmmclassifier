from collections import defaultdict
import numpy as np

def separate_data(X, lengths, labels):
    assert len(lengths) == len(labels), "Invalid dimensions"
    sequences = extract_sequence(X, lengths)
    grouped = defaultdict(list)
    for seq, label in zip(sequences, labels):
        grouped[label].append(seq)

    separated = {}
    for label, seqs in grouped.items():
        label_lengths = [len(s) for s in seqs]
        label_X = np.concatenate(seqs, axis=0)

        separated[label] = (label_X, label_lengths)

    return separated

def extract_sequence(X, lengths):
    assert sum(lengths) == len(X), "Invalid dimensions"
    sequences = []
    start = 0
    for l in lengths:
        end = start + l
        sequences.append(X[start:end])
        start = end
    return sequences


def bakis_components(n_components, skip):
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