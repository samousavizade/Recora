from dataclasses import dataclass
from typing import Generic, Iterable, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T", int, float)


@dataclass
class PairFeats(Generic[T]):
    user_feats: Optional[Iterable[T]]
    item_feats: Optional[Iterable[T]]


@dataclass
class TripleFeats(Generic[T]):
    query_feats: Optional[Iterable[T]]
    item_pos_feats: Optional[Iterable[T]]
    item_neg_feats: Optional[Iterable[T]]


@dataclass
class SeqFeats:
    interacted_seq: Iterable[Iterable[int]]
    interacted_len: Iterable[float]

    def repeat(self, num):
        self.interacted_seq = np.repeat(self.interacted_seq, num, axis=0)
        self.interacted_len = np.repeat(self.interacted_len, num)
        return self


@dataclass
class DualSeqFeats:
    long_seq: Iterable[Iterable[int]]
    long_len: Iterable[int]
    short_seq: Iterable[Iterable[int]]
    short_len: Iterable[int]


@dataclass
class SparseSeqFeats:
    interacted_indices: Iterable[Iterable[int]]
    interacted_values: Iterable[int]
    modified_batch_size: int


@dataclass
class PointwiseBatch:
    __slots__ = (
        "users",
        "items",
        "labels",
        "sample_weights",
        "sparse_indices",
        "dense_values",
        "seqs",
    )

    users: Iterable[int]
    items: Iterable[int]
    labels: Iterable[float]
    sample_weights: Iterable[float]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    seqs: Optional[SeqFeats]


@dataclass
class PointwiseSepFeatBatch(PointwiseBatch):
    sparse_indices: Optional[PairFeats[int]]
    dense_values: Optional[PairFeats[float]]


@dataclass
class PointwiseDualSeqBatch(PointwiseBatch):
    seqs: Optional[DualSeqFeats]


@dataclass
class PairwiseBatch:
    __slots__ = (
        "queries",
        "item_pairs",
        "sample_weights",
        "sparse_indices",
        "dense_values",
        "seqs",
    )

    queries: Iterable[int]
    item_pairs: Tuple[Iterable[int], Iterable[int]]
    sample_weights: Iterable[float]
    sparse_indices: Optional[TripleFeats[int]]
    dense_values: Optional[TripleFeats[float]]
    seqs: Optional[SeqFeats]


@dataclass
class SparseBatch:
    seqs: SparseSeqFeats
    items: Iterable[int]
    sample_weights: Iterable[float]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
