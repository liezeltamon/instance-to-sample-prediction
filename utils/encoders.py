from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import polars as pl


def concatenate_modalities(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("At least one modality array is required.")
    for array in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")
    return np.concatenate(arrays, axis=1)


class NumericEncoder:
    def __init__(self, feature_columns: Sequence[str]):
        self.feature_columns = feature_columns

    def fit_transform(self, table: pl.DataFrame) -> np.ndarray:
        return table.select(self.feature_columns).to_numpy().astype(np.float32)


class SequenceEncoder:
    def __init__(self, k: int = 2):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.vocabulary = self._build_vocab()

    def _build_vocab(self) -> list[str]:
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        return ["".join(p) for p in itertools.product(amino_acids, repeat=self.k)]

    def _kmers(self, sequence: str) -> Sequence[str]:
        return [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        n_samples = len(sequences)
        n_features = len(self.vocabulary)
        output = np.zeros((n_samples, n_features), dtype=np.float32)

        vocab_index = {kmer: idx for idx, kmer in enumerate(self.vocabulary)}
        for row_idx, sequence in enumerate(sequences):
            if sequence is None:
                continue
            for kmer in self._kmers(sequence):
                if kmer in vocab_index:
                    output[row_idx, vocab_index[kmer]] += 1.0
        return output

    def encode_dataframe(self, table: pl.DataFrame, column: str) -> np.ndarray:
        return self.encode(table[column].to_list())
