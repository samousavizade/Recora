import numpy as np

from .batch_unit import (
    DualSeqFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseDualSeqBatch,
    PointwiseSepFeatBatch,
    SparseBatch,
)
from .collators import merge_columns
from ..feature.ssl import get_ssl_features
from ..utils.constants import SequenceModels


def get_tf_feeds(model, data, is_training):
    if isinstance(data, SparseBatch):
        return _sparse_feed_dict(model, data, is_training)
    elif isinstance(data, PointwiseDualSeqBatch):
        return _dual_seq_feed_dict(model, data, is_training)
    elif isinstance(data, PointwiseSepFeatBatch):
        return _separate_feed_dict(model, data, is_training)
    elif isinstance(data, PairwiseBatch):
        return _pairwise_feed_dict(model, data, is_training)
    else:
        return _pointwise_feed_dict(model, data, is_training)


def _sparse_feed_dict(model, data: SparseBatch, is_training):
    feed_dict = {
        model.item_interaction_indices: data.seqs.interacted_indices,
        model.item_interaction_values: data.seqs.interacted_values,
        model.modified_batch_size: data.seqs.modified_batch_size,
        model.item_indices: data.items,
        model.is_training: is_training,
    }
    if hasattr(model, "sample_weights"):
        feed_dict.update({model.sample_weights: data.sample_weights})
    if hasattr(model, "user_sparse") and model.user_sparse:
        feed_dict.update({model.user_sparse_indices: data.sparse_indices})
    if hasattr(model, "user_dense") and model.user_dense:
        feed_dict.update({model.user_dense_values: data.dense_values})
    return feed_dict


def _maybe_add_graphsage_sampling_feeds(model, data, feed_dict):
    graph_data = getattr(data, "graph_data", None)
    if model.model_name != "GraphSage" or graph_data is None:
        return feed_dict

    feed_dict.update(
        {
            model.sampled_graph_indices: graph_data.graph_indices,
            model.sampled_graph_values: graph_data.graph_values,
            model.sampled_user_nodes: graph_data.sampled_user_nodes,
            model.sampled_item_nodes: graph_data.sampled_item_nodes,
            model.sampled_node_has_neighbors: graph_data.node_has_neighbors,
            model.sampled_batch_user_positions: graph_data.user_root_positions,
            model.sampled_batch_item_positions: graph_data.item_root_positions,
        }
    )
    if (
        hasattr(model, "sampled_batch_item_neg_positions")
        and graph_data.item_neg_root_positions is not None
    ):
        feed_dict.update(
            {model.sampled_batch_item_neg_positions: graph_data.item_neg_root_positions}
        )
    return feed_dict


def _pairwise_feed_dict(model, data: PairwiseBatch, is_training):
    if model.model_name == "BPR":
        feed_dict = {
            model.user_indices: data.queries,
            model.item_indices_pos: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
    elif model.model_name == "RNN4Rec":
        feed_dict = {
            model.user_interacted_seq: data.seqs.interacted_seq,
            model.user_interacted_len: data.seqs.interacted_len,
            model.item_indices_pos: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
    elif model.model_name == "TwoTower":
        feed_dict = {
            model.user_indices: data.queries,
            model.item_indices: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
        if model.user_sparse:
            feed_dict.update(
                {model.user_sparse_indices: data.sparse_indices.query_feats}
            )
        if model.user_dense:
            feed_dict.update({model.user_dense_values: data.dense_values.query_feats})
        if model.item_sparse:
            feed_dict.update(
                {model.item_sparse_indices: data.sparse_indices.item_pos_feats}
            )
            feed_dict.update(
                {model.item_sparse_indices_neg: data.sparse_indices.item_neg_feats}
            )
        if model.item_dense:
            feed_dict.update(
                {model.item_dense_values: data.dense_values.item_pos_feats}
            )
            feed_dict.update(
                {model.item_dense_values_neg: data.dense_values.item_neg_feats}
            )
    elif getattr(model, "separate_features", False) and model.loss_type == "max_margin":
        feed_dict = {
            model.user_indices: data.queries,
            model.item_indices: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
        if model.user_sparse:
            feed_dict.update(
                {model.user_sparse_indices: data.sparse_indices.query_feats}
            )
        if model.user_dense:
            feed_dict.update({model.user_dense_values: data.dense_values.query_feats})
        if data.seqs is not None:
            feed_dict.update(
                {
                    model.user_interacted_seq: data.seqs.interacted_seq,
                    model.user_interacted_len: data.seqs.interacted_len,
                }
            )
    elif getattr(model, "loss_type", None) in ("bpr", "ranknet", "lambdarank"):
        feed_dict = _generic_pairwise_feed_dict(model, data)
    else:
        raise ValueError(
            "Unsupported pairwise tf model. Expected legacy pairwise model or "
            "a model exposing `pairwise_logits`."
        )
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    if hasattr(model, "sample_weights"):
        feed_dict.update({model.sample_weights: data.sample_weights})
    return _maybe_add_graphsage_sampling_feeds(model, data, feed_dict)


def _pointwise_feed_dict(model, data: PointwiseBatch, is_training):
    feed_dict = dict()
    if hasattr(model, "user_indices"):
        feed_dict.update({model.user_indices: data.users})
    if hasattr(model, "item_indices"):
        feed_dict.update({model.item_indices: data.items})
    if hasattr(model, "labels"):
        feed_dict.update({model.labels: data.labels})
    if hasattr(model, "sample_weights"):
        feed_dict.update({model.sample_weights: data.sample_weights})
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: data.sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: data.dense_values})
    if SequenceModels.contains(model.model_name):
        feed_dict.update(
            {
                model.user_interacted_seq: data.seqs.interacted_seq,
                model.user_interacted_len: data.seqs.interacted_len,
            }
        )
    return _maybe_add_graphsage_sampling_feeds(model, data, feed_dict)


def _separate_feed_dict(model, data: PointwiseSepFeatBatch, is_training):
    feed_dict = {
        model.user_indices: data.users,
        model.item_indices: data.items,
        model.is_training: is_training,
    }
    if hasattr(model, "labels"):
        feed_dict.update({model.labels: data.labels})
    if hasattr(model, "sample_weights"):
        feed_dict.update({model.sample_weights: data.sample_weights})
    if hasattr(model, "correction"):
        feed_dict.update({model.correction: model.item_corrections[data.items]})
    if getattr(model, "user_sparse", False):
        feed_dict.update({model.user_sparse_indices: data.sparse_indices.user_feats})
    if getattr(model, "user_dense", False):
        feed_dict.update({model.user_dense_values: data.dense_values.user_feats})
    if getattr(model, "item_sparse", False):
        feed_dict.update({model.item_sparse_indices: data.sparse_indices.item_feats})
    if getattr(model, "item_dense", False):
        feed_dict.update({model.item_dense_values: data.dense_values.item_feats})
    if data.seqs is not None:
        feed_dict.update(
            {
                model.user_interacted_seq: data.seqs.interacted_seq,
                model.user_interacted_len: data.seqs.interacted_len,
            }
        )
    if hasattr(model, "ssl_pattern") and model.ssl_pattern is not None:
        ssl_feats = get_ssl_features(model, len(data.items))
        feed_dict.update(ssl_feats)
    return _maybe_add_graphsage_sampling_feeds(model, data, feed_dict)


def _dual_seq_feed_dict(model, data: PointwiseDualSeqBatch, is_training):
    feed_dict = {
        model.user_indices: data.users,
        model.item_indices: data.items,
        model.labels: data.labels,
        model.is_training: is_training,
        model.long_seqs: data.seqs.long_seq,
        model.long_seq_lens: data.seqs.long_len,
        model.short_seqs: data.seqs.short_seq,
        model.short_seq_lens: data.seqs.short_len,
    }
    if hasattr(model, "sample_weights"):
        feed_dict.update({model.sample_weights: data.sample_weights})
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: data.sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: data.dense_values})
    return _maybe_add_graphsage_sampling_feeds(model, data, feed_dict)


def _generic_pairwise_feed_dict(model, data: PairwiseBatch):
    queries = np.concatenate([data.queries, data.queries], axis=0)
    items = np.concatenate(data.item_pairs, axis=0)
    feed_dict = {
        model.user_indices: queries,
        model.item_indices: items,
    }

    if getattr(model, "sparse", False):
        sparse_pos, sparse_neg = _merge_pairwise_features(
            data.sparse_indices,
            model.data_info.user_sparse_col.index,
            model.data_info.item_sparse_col.index,
        )
        if sparse_pos is not None:
            feed_dict[model.sparse_indices] = np.concatenate(
                [sparse_pos, sparse_neg], axis=0
            )

    if getattr(model, "dense", False):
        dense_pos, dense_neg = _merge_pairwise_features(
            data.dense_values,
            model.data_info.user_dense_col.index,
            model.data_info.item_dense_col.index,
        )
        if dense_pos is not None:
            feed_dict[model.dense_values] = np.concatenate([dense_pos, dense_neg], axis=0)

    if data.seqs is not None:
        if isinstance(data.seqs, DualSeqFeats):
            feed_dict.update(
                {
                    model.long_seqs: np.concatenate(
                        [data.seqs.long_seq, data.seqs.long_seq], axis=0
                    ),
                    model.long_seq_lens: np.concatenate(
                        [data.seqs.long_len, data.seqs.long_len], axis=0
                    ),
                    model.short_seqs: np.concatenate(
                        [data.seqs.short_seq, data.seqs.short_seq], axis=0
                    ),
                    model.short_seq_lens: np.concatenate(
                        [data.seqs.short_len, data.seqs.short_len], axis=0
                    ),
                }
            )
        elif SequenceModels.contains(model.model_name):
            feed_dict.update(
                {
                    model.user_interacted_seq: np.concatenate(
                        [data.seqs.interacted_seq, data.seqs.interacted_seq], axis=0
                    ),
                    model.user_interacted_len: np.concatenate(
                        [data.seqs.interacted_len, data.seqs.interacted_len], axis=0
                    ),
                }
            )
    return feed_dict


def _merge_pairwise_features(triple_feats, user_col_index, item_col_index):
    if triple_feats is None:
        return None, None
    pos_feats = _merge_features(
        triple_feats.query_feats,
        triple_feats.item_pos_feats,
        user_col_index,
        item_col_index,
    )
    neg_feats = _merge_features(
        triple_feats.query_feats,
        triple_feats.item_neg_feats,
        user_col_index,
        item_col_index,
    )
    return pos_feats, neg_feats


def _merge_features(user_features, item_features, user_col_index, item_col_index):
    if user_features is not None and item_features is not None:
        return merge_columns(user_features, item_features, user_col_index, item_col_index)
    if user_features is not None:
        return user_features
    return item_features
