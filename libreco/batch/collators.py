import random

import numpy as np

from .batch_unit import (
    DualSeqFeats,
    PairFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseDualSeqBatch,
    PointwiseSepFeatBatch,
    SeqFeats,
    SparseBatch,
    SparseSeqFeats,
    TripleFeats,
)
from .enums import FeatType
from .sequence import get_dual_seqs, get_interacted_seqs, get_sparse_interacted
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
        if self.dual_seq:
            long_seqs, long_lens, short_seqs, short_lens = get_dual_seqs(
                user_indices,
                item_indices,
                self.user_consumed,
                self.n_items,
                self.long_max_len,
                self.short_max_len,
                self.user_consumed_set,
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
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
        )

    def get_seqs(self, user_indices, item_indices):
        if self.seq_mode == "random":
            self._set_random_seeds()
        batch_indices, batch_values, batch_size = get_sparse_interacted(
            user_indices,
            item_indices,
            self.user_consumed,
            self.seq_mode,
            self.max_seq_len,
            self.np_rng,
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
        else:
            users = batch["user"]
            items_pos = batch["item"]
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)

        sparse_batch = self.get_pairwise_feats(batch, FeatType.SPARSE, items_neg)
        dense_batch = self.get_pairwise_feats(batch, FeatType.DENSE, items_neg)
        seq_batch = self.get_seqs(users, items_pos)
        if self.has_seq and not self.repeat_positives and self.num_neg > 1:
            seq_batch = seq_batch.repeat(self.num_neg)
        return PairwiseBatch(
            queries=users,
            item_pairs=(items_pos, items_neg),
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
