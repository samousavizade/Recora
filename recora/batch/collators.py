import random
from dataclasses import replace

import numpy as np

from .batch_unit import (
    DualSeqFeats,
    PairFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseDualSeqBatch,
    PointwiseSepFeatBatch,
    SampledGraphData,
    SeqFeats,
    SparseBatch,
    SparseSeqFeats,
    TripleFeats,
)
from .enums import FeatType
from .sequence import get_dual_seqs, get_interacted_seqs, get_sparse_interacted
from ..graph import sample_bipartite_neighbors
from ..sampling import (
    neg_probs_from_frequency,
    negatives_from_popular,
    negatives_from_random,
    negatives_from_unconsumed,
)
from ..utils.constants import SequenceModels


def set_worker_seed(collator, seed):
    collator.worker_seed = seed
    collator.np_rng = None
    if hasattr(collator, "base_collator"):
        set_worker_seed(collator.base_collator, seed)


class BaseCollator:
    def __init__(self, model, data_info, separate_features=False, temperature=0.75):
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.item_consumed = data_info.item_consumed
        self.user_sparse_col_index = data_info.user_sparse_col.index
        self.item_sparse_col_index = data_info.item_sparse_col.index
        self.user_dense_col_index = data_info.user_dense_col.index
        self.item_dense_col_index = data_info.item_dense_col.index
        self.item_sparse_unique = data_info.item_sparse_unique
        self.item_dense_unique = data_info.item_dense_unique
        self.has_seq = True if SequenceModels.contains(model.model_name) else False
        self.seq_mode = model.seq_mode if hasattr(model, "seq_mode") else None
        self.max_seq_len = model.max_seq_len if hasattr(model, "max_seq_len") else None
        self.dual_seq = True if model.model_name == "SIM" else False
        self.long_max_len = model.long_max_len if self.dual_seq else None
        self.short_max_len = model.short_max_len if self.dual_seq else None
        self.separate_features = separate_features
        self.seed = model.seed
        self.worker_seed = model.seed
        self.temperature = temperature
        self.user_consumed_set = None
        self.neg_probs = None
        self.np_rng = None
        self.user_consumed_pos = None

    def __call__(self, batch):
        sparse_batch = self.get_features(batch, FeatType.SPARSE)
        dense_batch = self.get_features(batch, FeatType.DENSE)
        seq_batch = self.get_seqs(batch["user"], batch["item"])
        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch
        return batch_cls(
            users=batch["user"],
            items=batch["item"],
            labels=batch["label"],
            sample_weights=batch["sample_weight"],
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
        )

    def get_col_index(self, feat_type):
        if feat_type is FeatType.SPARSE:
            user_col_index = self.user_sparse_col_index
            item_col_index = self.item_sparse_col_index
        elif feat_type is FeatType.DENSE:
            user_col_index = self.user_dense_col_index
            item_col_index = self.item_dense_col_index
        else:
            raise ValueError("`feat_type` must be sparse or dense.")
        return user_col_index, item_col_index

    def get_features(self, batch, feat_type):
        if feat_type.value not in batch:
            return
        features = batch[feat_type.value]
        if self.separate_features:
            user_col_index, item_col_index = self.get_col_index(feat_type)
            user_features = features[:, user_col_index] if user_col_index else None
            item_features = features[:, item_col_index] if item_col_index else None
            features = PairFeats(user_features, item_features)
        return features

    def get_seqs(self, user_indices, item_indices):
        if not self.has_seq:
            return
        self._set_random_seeds()
        self._set_user_consumed()
        self._set_user_consumed_pos()
        if self.dual_seq:
            long_seqs, long_lens, short_seqs, short_lens = get_dual_seqs(
                user_indices,
                item_indices,
                self.user_consumed,
                self.n_items,
                self.long_max_len,
                self.short_max_len,
                self.user_consumed_set,
                self.user_consumed_pos,
            )
            return DualSeqFeats(long_seqs, long_lens, short_seqs, short_lens)
        seqs, seq_lens = get_interacted_seqs(
            user_indices,
            item_indices,
            self.user_consumed,
            self.n_items,
            self.seq_mode,
            self.max_seq_len,
            self.user_consumed_set,
            self.np_rng,
            self.user_consumed_pos,
        )
        return SeqFeats(seqs, seq_lens)

    def sample_neg_items(self, batch, sampler, num_neg):
        self._set_random_seeds()
        if sampler == "unconsumed":
            self._set_user_consumed()
            items_neg = negatives_from_unconsumed(
                self.user_consumed_set,
                batch["user"],
                batch["item"],
                self.n_items,
                num_neg,
            )
        elif sampler == "popular":
            self._set_neg_probs()
            items_neg = negatives_from_popular(
                self.np_rng,
                self.n_items,
                batch["item"],
                num_neg,
                probs=self.neg_probs,
            )
        else:
            items_neg = negatives_from_random(
                self.np_rng,
                self.n_items,
                batch["item"],
                num_neg,
            )
        return items_neg

    def _set_user_consumed(self):
        if self.user_consumed_set is None:
            self.user_consumed_set = [
                set(self.user_consumed[u]) for u in range(self.n_users)
            ]

    def _set_neg_probs(self):
        if self.neg_probs is None:
            self.neg_probs = neg_probs_from_frequency(
                self.item_consumed, self.n_items, self.temperature
            )

    def _set_user_consumed_pos(self):
        if self.user_consumed_pos is not None:
            return
        self.user_consumed_pos = []
        for u in range(self.n_users):
            consumed_items = self.user_consumed[u]
            position_map = {}
            for pos, item in enumerate(consumed_items):
                if item not in position_map:
                    position_map[item] = pos
            self.user_consumed_pos.append(position_map)

    def _set_random_seeds(self):
        if self.np_rng is None:
            random.seed(self.worker_seed)
            self.np_rng = np.random.default_rng(self.worker_seed)


class GraphSageSamplingCollator:
    def __init__(self, model, base_collator):
        self.base_collator = base_collator
        self.graph_neighbors = model.graph_neighbors
        self.n_users = model.n_users
        self.sample_sizes = tuple(model.sample_sizes)
        self.seed = model.seed
        self.worker_seed = model.seed
        self.np_rng = None

    def __call__(self, batch):
        base_batch = self.base_collator(batch)
        self._set_random_seeds()
        graph_data = self._sample_graph(base_batch)
        return replace(base_batch, graph_data=graph_data)

    def _sample_graph(self, batch):
        if isinstance(batch, PairwiseBatch):
            users = np.asarray(batch.queries, dtype=np.int32)
            items_pos = np.asarray(batch.item_pairs[0], dtype=np.int32)
            items_neg = np.asarray(batch.item_pairs[1], dtype=np.int32)
            item_roots = np.concatenate([items_pos, items_neg], axis=0)
        else:
            users = np.asarray(batch.users, dtype=np.int32)
            item_roots = np.asarray(batch.items, dtype=np.int32)
            items_neg = None

        (
            graph_indices,
            graph_values,
            sampled_user_nodes,
            sampled_item_nodes,
            node_has_neighbors,
            user_root_positions,
            item_root_positions,
        ) = sample_bipartite_neighbors(
            neighbors=self.graph_neighbors,
            n_users=self.n_users,
            user_roots=users,
            item_roots=item_roots,
            sample_sizes=self.sample_sizes,
            rng=self.np_rng,
        )

        item_neg_root_positions = None
        if items_neg is not None:
            item_neg_root_positions = item_root_positions[len(items_pos) :]
            item_root_positions = item_root_positions[: len(items_pos)]

        return SampledGraphData(
            graph_indices=graph_indices,
            graph_values=graph_values,
            sampled_user_nodes=sampled_user_nodes,
            sampled_item_nodes=sampled_item_nodes,
            node_has_neighbors=node_has_neighbors,
            user_root_positions=user_root_positions,
            item_root_positions=item_root_positions,
            item_neg_root_positions=item_neg_root_positions,
        )

    def _set_random_seeds(self):
        if self.np_rng is None:
            random.seed(self.worker_seed)
            self.np_rng = np.random.default_rng(self.worker_seed)


class SparseCollator(BaseCollator):
    def __call__(self, batch):
        seq_batch = self.get_seqs(batch["user"], batch["item"])
        sparse_batch = self.get_features(batch, FeatType.SPARSE)
        dense_batch = self.get_features(batch, FeatType.DENSE)
        return SparseBatch(
            seqs=seq_batch,
            items=batch["item"],
            sample_weights=batch["sample_weight"],
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
        )

    def get_seqs(self, user_indices, item_indices):
        if self.seq_mode == "random":
            self._set_random_seeds()
        self._set_user_consumed_pos()
        batch_indices, batch_values, batch_size = get_sparse_interacted(
            user_indices,
            item_indices,
            self.user_consumed,
            self.seq_mode,
            self.max_seq_len,
            self.np_rng,
            self.user_consumed_pos,
        )
        return SparseSeqFeats(batch_indices, batch_values, batch_size)


class PointwiseCollator(BaseCollator):
    def __init__(self, model, data_info, separate_features=False):
        super().__init__(model, data_info, separate_features)
        self.sampler = model.sampler
        self.num_neg = model.num_neg

    def __call__(self, batch):
        user_batch = np.repeat(batch["user"], self.num_neg + 1)
        item_batch = np.repeat(batch["item"], self.num_neg + 1)
        label_batch = np.zeros_like(item_batch, dtype=np.float32)
        sample_weight_batch = np.repeat(batch["sample_weight"], self.num_neg + 1)
        label_batch[:: (self.num_neg + 1)] = 1.0
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
        for i in range(self.num_neg):
            item_batch[(i + 1) :: (self.num_neg + 1)] = items_neg[i :: self.num_neg]

        sparse_batch = self.get_pointwise_feats(batch, FeatType.SPARSE, item_batch)
        dense_batch = self.get_pointwise_feats(batch, FeatType.DENSE, item_batch)
        seq_batch = self.get_seqs(user_batch, item_batch)
        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch
        return batch_cls(
            users=user_batch,
            items=item_batch,
            labels=label_batch,
            sample_weights=sample_weight_batch,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
        )

    def get_pointwise_feats(self, batch, feat_type, items):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        user_features = repeat_feats(batch_feats, user_col_index, self.num_neg)
        item_features = get_sampled_item_feats(self, item_col_index, items, feat_type)
        if self.separate_features:
            return PairFeats(user_features, item_features)
        if user_col_index and item_col_index:
            return merge_columns(
                user_features, item_features, user_col_index, item_col_index
            )
        return user_features if user_col_index else item_features


class SampledListwiseCollator(PointwiseCollator):
    def __init__(self, model, data_info, label_lookup, separate_features=False):
        super().__init__(model, data_info, separate_features)
        self.label_lookup = label_lookup
        self._label_row_cache = dict()
        self._positive_pool_cache = dict()
        self.listwise_num_pos = getattr(model, "listwise_num_pos", 1)
        if not isinstance(self.listwise_num_pos, int) or self.listwise_num_pos < 1:
            raise ValueError(
                "`listwise_num_pos` must be a positive integer for sampled listwise training"
            )
        self.list_size = self.listwise_num_pos + self.num_neg
        if self.num_neg < 1:
            raise ValueError(
                "`num_neg` must be positive for sampled listwise training"
            )
        if self.label_lookup is not None:
            self.label_lookup = self.label_lookup.tocsr().copy()
            self.label_lookup.sort_indices()

    def __call__(self, batch):
        self._set_random_seeds()
        users = np.asarray(batch["user"], dtype=np.int32)
        items_pos = np.asarray(batch["item"], dtype=np.int32)
        labels_pos = np.asarray(batch["label"], dtype=np.float32)
        batch_size = len(users)
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg).reshape(
            batch_size, self.num_neg
        )

        item_groups = np.empty((batch_size, self.list_size), dtype=np.int32)
        label_groups = np.zeros((batch_size, self.list_size), dtype=np.float32)
        item_groups[:, 0] = items_pos
        label_groups[:, 0] = labels_pos
        if self.listwise_num_pos > 1:
            extra_items, extra_labels = self._sample_extra_positive_groups(
                users, items_pos, labels_pos
            )
            item_groups[:, 1 : self.listwise_num_pos] = extra_items
            label_groups[:, 1 : self.listwise_num_pos] = extra_labels
        item_groups[:, self.listwise_num_pos :] = items_neg
        label_groups[:, self.listwise_num_pos :] = self._lookup_labels_batch(
            users, items_neg
        )

        user_batch = np.repeat(users, self.list_size)
        item_batch = item_groups.reshape(-1)
        label_batch = label_groups.reshape(-1)
        sample_weight_batch = np.repeat(batch["sample_weight"], self.list_size)

        sparse_batch = self.get_listwise_feats(batch, FeatType.SPARSE, item_batch)
        dense_batch = self.get_listwise_feats(batch, FeatType.DENSE, item_batch)
        seq_batch = self.get_seqs(user_batch, item_batch)
        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch
        return batch_cls(
            users=user_batch,
            items=item_batch,
            labels=label_batch,
            sample_weights=sample_weight_batch,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
        )

    def _sample_extra_positive_groups(self, users, anchor_items, anchor_labels):
        extra_size = self.listwise_num_pos - 1
        extra_items = np.repeat(anchor_items[:, None], extra_size, axis=1).astype(
            np.int32, copy=False
        )
        extra_labels = np.repeat(anchor_labels[:, None], extra_size, axis=1).astype(
            np.float32, copy=False
        )
        for user, row_positions in self._iter_user_row_positions(users):
            pool_items = self._positive_item_pool(user)
            if pool_items.size == 0:
                continue

            user_anchors = anchor_items[row_positions]
            user_extra_items = np.empty((len(row_positions), extra_size), dtype=np.int32)
            for i, anchor_item in enumerate(user_anchors):
                valid_pool = pool_items[pool_items != anchor_item]
                if valid_pool.size == 0:
                    user_extra_items[i].fill(anchor_item)
                    continue
                sampled_items = self.np_rng.choice(
                    valid_pool, size=extra_size, replace=(len(valid_pool) < extra_size)
                )
                user_extra_items[i] = sampled_items.astype(np.int32, copy=False)

            extra_items[row_positions] = user_extra_items
            flat_items = user_extra_items.reshape(-1)
            if self.label_lookup is None:
                flat_labels = self._lookup_labels(user, flat_items)
            else:
                row_indices, row_data = self._get_label_row(user)
                flat_labels = self._lookup_labels_with_row(
                    row_indices, row_data, flat_items
                )
            extra_labels[row_positions] = flat_labels.reshape(len(row_positions), extra_size)
        return extra_items, extra_labels

    def _positive_item_pool(self, user):
        user = int(user)
        if user in self._positive_pool_cache:
            return self._positive_pool_cache[user]
        if self.label_lookup is not None:
            row_indices, row_data = self._get_label_row(user)
            if row_indices is None:
                pool_items = np.empty(0, dtype=np.int32)
            else:
                pool_items = row_indices[row_data > 0].astype(np.int32)
        else:
            pool_items = np.unique(np.asarray(self.user_consumed[user], dtype=np.int32))
        self._positive_pool_cache[user] = pool_items
        return pool_items

    def _lookup_labels(self, user, items):
        row_indices, row_data = self._get_label_row(user)
        return self._lookup_labels_with_row(row_indices, row_data, items)

    def _lookup_labels_batch(self, users, items):
        labels = np.zeros_like(items, dtype=np.float32)
        if self.label_lookup is None or items.size == 0:
            return labels
        for user, row_positions in self._iter_user_row_positions(users):
            row_indices, row_data = self._get_label_row(user)
            if row_indices is None:
                continue
            user_items = items[row_positions].reshape(-1)
            user_labels = self._lookup_labels_with_row(row_indices, row_data, user_items)
            labels[row_positions] = user_labels.reshape(len(row_positions), items.shape[1])
        return labels

    @staticmethod
    def _iter_user_row_positions(users):
        unique_users, inverse = np.unique(users, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
        grouped_positions = np.split(order, boundaries)
        return zip(unique_users, grouped_positions)

    def _get_label_row(self, user):
        user = int(user)
        if self.label_lookup is None:
            return None, None
        if user in self._label_row_cache:
            return self._label_row_cache[user]
        row = self.label_lookup.getrow(user)
        if row.nnz == 0:
            cached = (None, None)
        else:
            cached = (
                row.indices.astype(np.int32, copy=False),
                row.data.astype(np.float32, copy=False),
            )
        self._label_row_cache[user] = cached
        return cached

    @staticmethod
    def _lookup_labels_with_row(row_indices, row_data, items):
        labels = np.zeros(len(items), dtype=np.float32)
        if row_indices is None or len(items) == 0:
            return labels
        positions = np.searchsorted(row_indices, items)
        valid = positions < len(row_indices)
        if not np.any(valid):
            return labels
        valid_indices = np.flatnonzero(valid)
        valid_positions = positions[valid_indices]
        matched = row_indices[valid_positions] == items[valid_indices]
        if np.any(matched):
            labels[valid_indices[matched]] = row_data[valid_positions[matched]]
        return labels

    def get_listwise_feats(self, batch, feat_type, items):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        user_features = None
        if user_col_index:
            user_features = np.repeat(
                batch_feats[:, user_col_index], self.list_size, axis=0
            )
        item_features = get_sampled_item_feats(self, item_col_index, items, feat_type)
        if self.separate_features:
            return PairFeats(user_features, item_features)
        if user_col_index and item_col_index:
            return merge_columns(
                user_features, item_features, user_col_index, item_col_index
            )
        return user_features if user_col_index else item_features


class PairwiseCollator(BaseCollator):
    def __init__(self, model, data_info, repeat_positives):
        super().__init__(model, data_info, separate_features=True)
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        self.repeat_positives = repeat_positives

    def __call__(self, batch):
        if self.repeat_positives and self.num_neg > 1:
            users = np.repeat(batch["user"], self.num_neg)
            items_pos = np.repeat(batch["item"], self.num_neg)
            sample_weights = np.repeat(batch["sample_weight"], self.num_neg)
        else:
            users = batch["user"]
            items_pos = batch["item"]
            sample_weights = batch["sample_weight"]
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)

        sparse_batch = self.get_pairwise_feats(batch, FeatType.SPARSE, items_neg)
        dense_batch = self.get_pairwise_feats(batch, FeatType.DENSE, items_neg)
        seq_batch = self.get_seqs(users, items_pos)
        if self.has_seq and not self.repeat_positives and self.num_neg > 1:
            seq_batch = seq_batch.repeat(self.num_neg)
        return PairwiseBatch(
            queries=users,
            item_pairs=(items_pos, items_neg),
            sample_weights=sample_weights,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
        )

    def get_pairwise_feats(self, batch, feat_type, items_neg):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        if self.repeat_positives and self.num_neg > 1:
            user_feats = repeat_feats(
                batch_feats, user_col_index, self.num_neg, is_pairwise=True
            )
            item_pos_feats = repeat_feats(
                batch_feats, item_col_index, self.num_neg, is_pairwise=True
            )
        else:
            user_feats = batch_feats[:, user_col_index] if user_col_index else None
            item_pos_feats = batch_feats[:, item_col_index] if item_col_index else None
        item_neg_feats = get_sampled_item_feats(
            self, item_col_index, items_neg, feat_type
        )
        return TripleFeats(user_feats, item_pos_feats, item_neg_feats)


def repeat_feats(batch_feats, col_index, num_neg, is_pairwise=False):
    if not col_index:
        return
    column_features = batch_feats[:, col_index]
    repeats = num_neg if is_pairwise else num_neg + 1
    return np.repeat(column_features, repeats, axis=0)


def get_sampled_item_feats(collator, item_col_index, items_sampled, feat_type):
    if not item_col_index:
        return
    if feat_type is FeatType.SPARSE:
        item_unique_features = collator.item_sparse_unique
    elif feat_type is FeatType.DENSE:
        item_unique_features = collator.item_dense_unique
    else:
        raise ValueError("`feat_type` must be sparse or dense.")
    return item_unique_features[items_sampled]


def merge_columns(user_features, item_features, user_col_index, item_col_index):
    if len(user_features) != len(item_features):
        raise ValueError(
            f"length of user_features and length of item_features don't match, "
            f"got {len(user_features)} and {len(item_features)}"
        )
    orig_cols = user_col_index + item_col_index
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
    concat_features = np.concatenate([user_features, item_features], axis=1)
    return concat_features[:, col_reindex]
