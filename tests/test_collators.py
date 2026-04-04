from io import StringIO

import numpy as np
import pandas as pd
import pytest

from recora.algorithms import DIN, RNN4Rec, TwoTower
from recora.batch.batch_data import BatchData
from recora.batch.batch_unit import (
    PairFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseSepFeatBatch,
    SparseBatch,
    TripleFeats,
)
from recora.batch.collators import BaseCollator as NormalCollator
from recora.batch.collators import PairwiseCollator, PointwiseCollator, SparseCollator
from recora.data import DatasetFeat
from recora.sampling.negatives import negatives_from_unconsumed
from recora.tfops import tf

raw_data = """
user,item,label,time,sex,age,occupation,genre1,genre2,genre3
1,296,2,964138229,F,25,6,crime,drama,missing
1,297,2,964138229,F,25,6,crime,drama,missing
1298,208,4,974849526,M,35,6,action,adventure,missing
2,1769,4,964322774,M,35,7,action,thriller,missing
1298,933,4,974607346,M,45,6,romance,missing,missing
3706,1136,5,966376465,M,25,12,comedy,missing,missing
2137,1215,3,974640099,F,1,10,action,adventure,comedy
2,1257,4,974170662,M,18,4,comedy,missing,missing
242,3148,3,977854274,F,18,4,drama,missing,missing
2211,932,4,974607346,M,45,6,romance,missing,missing
263,2115,2,976651827,F,25,7,action,adventure,missing
1,291,2,964138229,F,25,6,crime,drama,missing
5184,866,5,961735308,M,18,20,crime,drama,romance
"""


@pytest.fixture
def config_feat_data(request):
    pd_data = pd.read_csv(StringIO(raw_data), sep=",", header=0)
    pd_data["item_dense_col"] = np.random.default_rng(42).random(len(pd_data))
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=pd_data, **request.param
    )
    return train_data, data_info


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex"],
            "dense_col": ["item_dense_col"],
            "user_col": ["sex"],
            "item_col": ["item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_normal_collator(config_feat_data):
    train_data, data_info = config_feat_data
    din = DIN("ranking", data_info, sampler=None)
    two_tower = TwoTower("ranking", data_info, loss_type="softmax", sampler=None)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    normal_collator = NormalCollator(din, data_info, separate_features=False)
    normal_batch = normal_collator(original_data)
    assert isinstance(normal_batch, PointwiseBatch)
    assert len(normal_batch.labels) == 3
    assert isinstance(normal_batch.items, np.ndarray)
    if normal_batch.sparse_indices is not None:
        assert isinstance(normal_batch.sparse_indices, np.ndarray)
        assert normal_batch.sparse_indices.shape[1] == len(data_info.sparse_col.index)
    if normal_batch.dense_values is not None:
        assert isinstance(normal_batch.dense_values, np.ndarray)
        assert normal_batch.dense_values.shape[1] == len(data_info.dense_col.index)
    assert isinstance(normal_batch.seqs.interacted_seq, np.ndarray)
    assert normal_batch.seqs.interacted_seq.shape == (3, 10)
    assert isinstance(normal_batch.seqs.interacted_len, np.ndarray)
    assert normal_batch.seqs.interacted_len.shape == (3,)
    normal_batch.seqs.repeat(3)
    assert normal_batch.seqs.interacted_seq.shape == (9, 10)
    assert normal_batch.seqs.interacted_len.shape == (9,)
    tf.reset_default_graph()

    sep_collator = NormalCollator(two_tower, data_info, separate_features=True)
    sep_batch = sep_collator(original_data)
    assert isinstance(sep_batch, PointwiseSepFeatBatch)
    assert len(sep_batch.labels) == 3
    if sep_batch.sparse_indices is not None:
        assert isinstance(sep_batch.sparse_indices, PairFeats)
    user_sparse_len = len(data_info.user_sparse_col.index)
    if user_sparse_len > 0:
        assert isinstance(sep_batch.sparse_indices.user_feats, np.ndarray)
        assert sep_batch.sparse_indices.user_feats.shape[1] == user_sparse_len
    item_sparse_len = len(data_info.item_sparse_col.index)
    if item_sparse_len > 0:
        assert isinstance(sep_batch.sparse_indices.item_feats, np.ndarray)
        assert sep_batch.sparse_indices.item_feats.shape[1] == item_sparse_len
    user_dense_len = len(data_info.user_dense_col.index)
    if user_dense_len > 0:
        assert isinstance(sep_batch.dense_values.user_feats, np.ndarray)
        assert sep_batch.dense_values.user_feats.shape[1] == user_dense_len
    item_dense_len = len(data_info.item_dense_col.index)
    if item_dense_len > 0:
        assert isinstance(sep_batch.dense_values.item_feats, np.ndarray)
        assert sep_batch.dense_values.item_feats.shape[1] == item_dense_len
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_sparse_collator(config_feat_data):
    train_data, data_info = config_feat_data
    model = DIN("ranking", data_info, sampler=None)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = SparseCollator(model, data_info)
    batch = collator(original_data)
    assert isinstance(batch, SparseBatch)
    assert len(batch.items) == 3
    assert isinstance(batch.items, np.ndarray)
    if batch.sparse_indices is not None:
        assert isinstance(batch.sparse_indices, np.ndarray)
        assert batch.sparse_indices.shape[1] == len(data_info.sparse_col.index)
    if batch.dense_values is not None:
        assert isinstance(batch.dense_values, np.ndarray)
        assert batch.dense_values.shape[1] == len(data_info.dense_col.index)
    assert isinstance(batch.seqs.interacted_indices, np.ndarray)
    assert batch.seqs.interacted_indices.shape == (3, 2)
    assert isinstance(batch.seqs.interacted_values, np.ndarray)
    assert batch.seqs.interacted_values.shape == (3,)
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_pointwise_collator(config_feat_data):
    train_data, data_info = config_feat_data
    model = DIN("ranking", data_info, "cross_entropy", sampler="random", num_neg=2)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = PointwiseCollator(model, data_info)
    batch = collator(original_data)
    assert isinstance(batch, PointwiseBatch)
    assert len(batch.users) == len(batch.items) == len(batch.labels) == 9
    assert isinstance(batch.items, np.ndarray)
    assert np.all(batch.labels[0::3] == 1.0)
    assert np.all(batch.labels[1::3] == 0.0)
    assert np.all(batch.labels[2::3] == 0.0)
    if batch.sparse_indices is not None:
        assert isinstance(batch.sparse_indices, np.ndarray)
        assert batch.sparse_indices.shape[1] == len(data_info.sparse_col.index)
    if batch.dense_values is not None:
        assert isinstance(batch.dense_values, np.ndarray)
        assert batch.dense_values.shape[1] == len(data_info.dense_col.index)
    assert isinstance(batch.seqs.interacted_seq, np.ndarray)
    assert batch.seqs.interacted_seq.shape == (9, 10)
    assert isinstance(batch.seqs.interacted_len, np.ndarray)
    assert batch.seqs.interacted_len.shape == (9,)
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_pairwise_sequence_collator(config_feat_data):
    train_data, data_info = config_feat_data
    model = RNN4Rec("ranking", data_info, "bpr", num_neg=2)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = PairwiseCollator(model, data_info, repeat_positives=True)
    batch = collator(original_data)
    assert isinstance(batch, PairwiseBatch)
    assert len(batch.queries) == 6
    assert len(batch.item_pairs[0]) == len(batch.item_pairs[1]) == 6
    assert isinstance(batch.queries, np.ndarray)
    assert np.all(batch.queries[:2] == batch.queries[0])
    assert np.all(batch.item_pairs[0][:2] == batch.item_pairs[0][:1])
    assert batch.sparse_indices is None
    assert batch.dense_values is None
    assert isinstance(batch.seqs.interacted_seq, np.ndarray)
    assert batch.seqs.interacted_seq.shape == (6, 10)
    assert isinstance(batch.seqs.interacted_len, np.ndarray)
    assert batch.seqs.interacted_len.shape == (6,)
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_pairwise_separate_features_collator(config_feat_data):
    train_data, data_info = config_feat_data
    model = TwoTower("ranking", data_info, "max_margin", num_neg=3)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = PairwiseCollator(model, data_info, repeat_positives=True)
    batch = collator(original_data)
    assert isinstance(batch, PairwiseBatch)
    assert len(batch.queries) == len(batch.item_pairs[0]) == len(batch.item_pairs[1]) == 9
    assert isinstance(batch.queries, np.ndarray)
    assert batch.seqs is None
    assert isinstance(batch.sparse_indices, TripleFeats) or batch.sparse_indices is None
    assert isinstance(batch.dense_values, TripleFeats) or batch.dense_values is None
    user_sparse_len = len(data_info.user_sparse_col.index)
    if user_sparse_len > 0:
        assert isinstance(batch.sparse_indices.query_feats, np.ndarray)
        assert batch.sparse_indices.query_feats.shape[1] == user_sparse_len
    item_sparse_len = len(data_info.item_sparse_col.index)
    if item_sparse_len > 0:
        assert isinstance(batch.sparse_indices.item_pos_feats, np.ndarray)
        assert batch.sparse_indices.item_pos_feats.shape[1] == item_sparse_len
        assert isinstance(batch.sparse_indices.item_neg_feats, np.ndarray)
        assert batch.sparse_indices.item_neg_feats.shape[1] == item_sparse_len
    user_dense_len = len(data_info.user_dense_col.index)
    if user_dense_len > 0:
        assert isinstance(batch.dense_values.query_feats, np.ndarray)
        assert batch.dense_values.query_feats.shape[1] == user_dense_len
    item_dense_len = len(data_info.item_dense_col.index)
    if item_dense_len > 0:
        assert isinstance(batch.dense_values.item_pos_feats, np.ndarray)
        assert batch.dense_values.item_pos_feats.shape[1] == item_dense_len
        assert isinstance(batch.dense_values.item_neg_feats, np.ndarray)
        assert batch.dense_values.item_neg_feats.shape[1] == item_dense_len
    tf.reset_default_graph()


def test_negatives_exceed_sampling_tolerance():
    users = [0, 1, 2]
    items = [1, 2, 4]
    user_consumed_set = {0: {1}, 1: {3, 4}, 2: {1, 2, 3}}
    n_items = 5
    num_neg = 5
    tolerance = 100
    negatives = np.array_split(
        negatives_from_unconsumed(
            user_consumed_set, users, items, n_items, num_neg, tolerance
        ),
        3,
    )
    assert 1 not in negatives[0][:4]
    assert 2 not in negatives[1][:4]
    assert 4 not in negatives[2][:4]
