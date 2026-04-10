from io import StringIO

import numpy as np
import pandas as pd
import pytest

from recora.algorithms import DIN, FM, RNN4Rec, SIM, SVD, TwoTower
from recora.batch import get_tf_feeds
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
from recora.batch.collators import (
    PairwiseCollator,
    PointwiseCollator,
    SparseCollator,
    merge_columns,
)
from recora.data import DatasetFeat
from recora.sampling.negatives import (
    negatives_from_popular_unconsumed,
    negatives_from_unconsumed,
)
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
    pd_data["sample_weight"] = np.arange(1, len(pd_data) + 1, dtype=np.float32)
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
    assert np.array_equal(normal_batch.sample_weights, train_data.sample_weights[[11, 7, 2]])
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
    assert np.array_equal(sep_batch.sample_weights, train_data.sample_weights[[11, 7, 2]])
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
    assert np.array_equal(batch.sample_weights, train_data.sample_weights[[11, 7, 2]])
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
    assert np.array_equal(
        batch.sample_weights, np.repeat(train_data.sample_weights[[11, 7, 2]], 3)
    )
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
    assert np.array_equal(
        batch.sample_weights, np.repeat(train_data.sample_weights[[11, 7, 2]], 2)
    )
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
    assert np.array_equal(
        batch.sample_weights, np.repeat(train_data.sample_weights[[11, 7, 2]], 3)
    )
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


@pytest.mark.parametrize("loss_type", ["bpr", "lambdarank"])
def test_pairwise_feed_dict_pure_model(pure_data_small, loss_type):
    _, train_data, _, data_info = pure_data_small
    model = SVD("ranking", data_info, loss_type=loss_type, num_neg=2)
    model.build_model()
    original_data = BatchData(train_data, use_features=False)[[0, 1, 2]]

    collator = PairwiseCollator(model, data_info, repeat_positives=True)
    batch = collator(original_data)
    feed_dict = get_tf_feeds(model, batch, is_training=True)

    expected_queries = np.concatenate([batch.queries, batch.queries], axis=0)
    expected_items = np.concatenate(batch.item_pairs, axis=0)
    assert np.array_equal(feed_dict[model.user_indices], expected_queries)
    assert np.array_equal(feed_dict[model.item_indices], expected_items)
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("loss_type", ["ranknet", "lambdarank"])
def test_pairwise_feed_dict_merged_features(config_feat_data, loss_type):
    train_data, data_info = config_feat_data
    model = FM("ranking", data_info, loss_type, num_neg=2)
    model.build_model()
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = PairwiseCollator(model, data_info, repeat_positives=True)
    batch = collator(original_data)
    feed_dict = get_tf_feeds(model, batch, is_training=True)

    expected_queries = np.concatenate([batch.queries, batch.queries], axis=0)
    expected_items = np.concatenate(batch.item_pairs, axis=0)
    expected_sparse_pos = merge_columns(
        batch.sparse_indices.query_feats,
        batch.sparse_indices.item_pos_feats,
        data_info.user_sparse_col.index,
        data_info.item_sparse_col.index,
    )
    expected_sparse_neg = merge_columns(
        batch.sparse_indices.query_feats,
        batch.sparse_indices.item_neg_feats,
        data_info.user_sparse_col.index,
        data_info.item_sparse_col.index,
    )
    expected_dense_pos = merge_columns(
        batch.dense_values.query_feats,
        batch.dense_values.item_pos_feats,
        data_info.user_dense_col.index,
        data_info.item_dense_col.index,
    )
    expected_dense_neg = merge_columns(
        batch.dense_values.query_feats,
        batch.dense_values.item_neg_feats,
        data_info.user_dense_col.index,
        data_info.item_dense_col.index,
    )

    assert np.array_equal(feed_dict[model.user_indices], expected_queries)
    assert np.array_equal(feed_dict[model.item_indices], expected_items)
    assert np.array_equal(
        feed_dict[model.sparse_indices],
        np.concatenate([expected_sparse_pos, expected_sparse_neg], axis=0),
    )
    assert np.array_equal(
        feed_dict[model.dense_values],
        np.concatenate([expected_dense_pos, expected_dense_neg], axis=0),
    )
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        }
    ],
    indirect=True,
)
def test_pairwise_feed_dict_sim_dual_sequences(config_feat_data):
    train_data, data_info = config_feat_data
    model = SIM(
        "ranking",
        data_info,
        "ranknet",
        num_neg=2,
        search_topk=1,
        long_max_len=4,
        short_max_len=2,
    )
    model.build_model()
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = PairwiseCollator(model, data_info, repeat_positives=True)
    batch = collator(original_data)
    feed_dict = get_tf_feeds(model, batch, is_training=True)
    split_index = len(batch.queries)

    assert np.array_equal(feed_dict[model.user_indices][:split_index], batch.queries)
    assert np.array_equal(feed_dict[model.user_indices][split_index:], batch.queries)
    assert np.array_equal(feed_dict[model.long_seqs][:split_index], batch.seqs.long_seq)
    assert np.array_equal(feed_dict[model.long_seqs][split_index:], batch.seqs.long_seq)
    assert np.array_equal(
        feed_dict[model.short_seqs][:split_index], batch.seqs.short_seq
    )
    assert np.array_equal(
        feed_dict[model.short_seqs][split_index:], batch.seqs.short_seq
    )
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


def test_popular_unconsumed_negatives_exclude_consumed_and_positive():
    np_rng = np.random.default_rng(42)
    users = np.array([0, 1, 2], dtype=np.int32)
    items = np.array([1, 2, 4], dtype=np.int32)
    user_consumed_set = {0: {1, 2}, 1: {2, 3}, 2: {1, 2, 3, 4}}
    n_items = 6
    num_neg = 4
    probs = np.array([0.50, 0.20, 0.10, 0.08, 0.07, 0.05], dtype=np.float64)
    probs /= np.sum(probs)
    negatives = np.array_split(
        negatives_from_popular_unconsumed(
            np_rng=np_rng,
            user_consumed_set=user_consumed_set,
            users=users,
            items_pos=items,
            n_items=n_items,
            num_neg=num_neg,
            probs=probs,
            tolerance=10,
        ),
        len(users),
    )
    for user, item, user_negs in zip(users, items, negatives):
        assert item not in user_negs
        assert not set(user_negs).intersection(user_consumed_set[int(user)])


def test_popular_unconsumed_negatives_are_deterministic_with_seed():
    users = np.array([0, 0, 1, 1], dtype=np.int32)
    items = np.array([1, 2, 1, 3], dtype=np.int32)
    user_consumed_set = {0: {1, 2}, 1: {1, 3}}
    probs = np.array([0.4, 0.2, 0.2, 0.1, 0.1], dtype=np.float64)
    probs /= np.sum(probs)
    negatives_1 = negatives_from_popular_unconsumed(
        np_rng=np.random.default_rng(7),
        user_consumed_set=user_consumed_set,
        users=users,
        items_pos=items,
        n_items=5,
        num_neg=3,
        probs=probs,
    )
    negatives_2 = negatives_from_popular_unconsumed(
        np_rng=np.random.default_rng(7),
        user_consumed_set=user_consumed_set,
        users=users,
        items_pos=items,
        n_items=5,
        num_neg=3,
        probs=probs,
    )
    assert np.array_equal(negatives_1, negatives_2)


def test_popular_unconsumed_preserves_popularity_bias():
    np_rng = np.random.default_rng(11)
    users = np.zeros(6000, dtype=np.int32)
    items = np.ones(6000, dtype=np.int32)
    user_consumed_set = {0: {1}}
    probs = np.array([0.03, 0.02, 0.10, 0.20, 0.65], dtype=np.float64)
    probs /= np.sum(probs)
    negatives = negatives_from_popular_unconsumed(
        np_rng=np_rng,
        user_consumed_set=user_consumed_set,
        users=users,
        items_pos=items,
        n_items=5,
        num_neg=1,
        probs=probs,
    )
    counts = np.bincount(negatives, minlength=5)
    assert counts[1] == 0
    assert counts[4] > counts[3] > counts[2] > counts[0]


def test_popular_unconsumed_dense_user_exact_fallback():
    np_rng = np.random.default_rng(1)
    users = np.zeros(128, dtype=np.int32)
    items = np.full(128, 3, dtype=np.int32)
    user_consumed_set = {0: set(range(9))}
    n_items = 10
    probs = np.arange(1, n_items + 1, dtype=np.float64)
    probs /= np.sum(probs)
    negatives = negatives_from_popular_unconsumed(
        np_rng=np_rng,
        user_consumed_set=user_consumed_set,
        users=users,
        items_pos=items,
        n_items=n_items,
        num_neg=2,
        probs=probs,
        tolerance=1,
    )
    assert np.all(negatives == 9)


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("sampler", ["popular_unconsumed", "popular+unconsumed"])
def test_pointwise_collator_popular_unconsumed_sampler(config_feat_data, sampler):
    train_data, data_info = config_feat_data
    model = DIN("ranking", data_info, "cross_entropy", sampler=sampler, num_neg=2)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    collator = PointwiseCollator(model, data_info)
    batch = collator(original_data)
    group_size = model.num_neg + 1
    item_groups = batch.items.reshape(len(original_data["item"]), group_size)
    user_groups = batch.users.reshape(len(original_data["item"]), group_size)
    for group_items, group_users in zip(item_groups, user_groups):
        pos_item = int(group_items[0])
        neg_items = group_items[1:]
        user = int(group_users[0])
        consumed = set(data_info.user_consumed[user])
        assert pos_item not in neg_items
        assert not set(neg_items).intersection(consumed)
    tf.reset_default_graph()
