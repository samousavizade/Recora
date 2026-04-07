import abc

import numpy as np

from .dyn_embed_base import DynEmbedBase
from ..graph import build_tf_sparse_tensor
from ..layers import dense_nn, normalize_embeds
from ..tfops import dropout_config, rebuild_tf_model, reg_config, tf
from ..tfops.features import compute_sparse_feats, get_feed_dict
from ..utils.misc import hidden_units_config
from ..utils.validate import (
    check_multi_sparse,
    dense_field_size,
    sparse_feat_size,
)


class GraphFeatBase(DynEmbedBase):
    user_variables = ("embedding/user_embeds_var",)
    item_variables = ("embedding/item_embeds_var",)
    sparse_variables = ("embedding/sparse_embeds_var",)
    dense_variables = ("embedding/dense_embeds_var",)

    def __init__(
        self,
        task,
        data_info,
        embed_size,
        norm_embed,
        hidden_units,
        layer_sizes,
        use_bn,
        dropout_rate,
        reg,
        recent_num,
        random_num,
        multi_sparse_combiner,
        tf_sess_config,
        use_correction,
    ):
        super().__init__(
            task,
            data_info,
            embed_size,
            norm_embed,
            recent_num=recent_num,
            random_num=random_num,
            tf_sess_config=tf_sess_config,
        )
        self.hidden_units = hidden_units_config(hidden_units)
        self.layer_sizes = hidden_units_config(layer_sizes)
        self.embedding_dim = self.layer_sizes[-1]
        self.query_hidden_units = [*self.hidden_units, self.embedding_dim]
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.reg = reg_config(reg)
        self.use_correction = use_correction
        self.margin = 1.0
        self.separate_features = True
        self.user_sparse = bool(data_info.user_sparse_col.name)
        self.user_dense = bool(data_info.user_dense_col.name)
        self.item_sparse = False
        self.item_dense = False
        self.has_item_sparse_feats = bool(data_info.item_sparse_col.name)
        self.has_item_dense_feats = bool(data_info.item_dense_col.name)
        self.user_dense_field_size = len(data_info.user_dense_col.name)
        self.item_dense_field_size = len(data_info.item_dense_col.name)
        if self.user_sparse or self.has_item_sparse_feats:
            self.sparse_feature_size = sparse_feat_size(data_info)
            self.multi_sparse_combiner = check_multi_sparse(
                data_info, multi_sparse_combiner
            )
        if self.user_dense or self.has_item_dense_feats:
            self.dense_field_size = dense_field_size(data_info)

    def build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        if self.loss_type in ("cross_entropy", "listnet", "approx_ndcg"):
            self.labels = tf.placeholder(tf.float32, shape=[None])
        if self.loss_type == "max_margin":
            self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])
        if self.loss_type == "softmax" and self.use_correction:
            self.correction = tf.placeholder(tf.float32, shape=[None])
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.max_seq_len]
        )
        self.user_interacted_len = tf.placeholder(tf.int32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        if self.user_sparse:
            self.user_sparse_indices = tf.placeholder(
                tf.int32, shape=[None, len(self.data_info.user_sparse_col.name)]
            )
        if self.user_dense:
            self.user_dense_values = tf.placeholder(
                tf.float32, shape=[None, len(self.data_info.user_dense_col.name)]
            )

    def build_variables(self):
        with tf.variable_scope("embedding"):
            self.user_embeds_var = tf.get_variable(
                name="user_embeds_var",
                shape=(self.n_users + 1, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.item_embeds_var = tf.get_variable(
                name="item_embeds_var",
                shape=(self.n_items, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            if self.user_sparse or self.has_item_sparse_feats:
                self.sparse_embeds_var = tf.get_variable(
                    name="sparse_embeds_var",
                    shape=(self.sparse_feature_size, self.embed_size),
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
            if self.user_dense or self.has_item_dense_feats:
                self.dense_embeds_var = tf.get_variable(
                    name="dense_embeds_var",
                    shape=(self.dense_field_size, self.embed_size),
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )

    def compute_user_base_embeddings(
        self, user_indices, sparse_indices=None, dense_values=None, reuse_layer=False
    ):
        user_embed = tf.nn.embedding_lookup(self.user_embeds_var, user_indices)
        concat_embeds = [user_embed]
        if sparse_indices is not None:
            sparse_embed = compute_sparse_feats(
                self.data_info,
                self.multi_sparse_combiner,
                sparse_indices,
                var_name="sparse_embeds_var",
                var_shape=(self.sparse_feature_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                reuse_layer=reuse_layer,
                flatten=True,
            )
            concat_embeds.append(sparse_embed)
        if dense_values is not None:
            dense_embed = self._compute_dense_feats(
                dense_values,
                self.data_info.user_dense_col.index,
            )
            concat_embeds.append(dense_embed)
        user_inputs = (
            tf.concat(concat_embeds, axis=1)
            if len(concat_embeds) > 1
            else concat_embeds[0]
        )
        return dense_nn(
            user_inputs,
            [*self.hidden_units, self.embed_size],
            activation=tf.nn.leaky_relu,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            reuse_layer=reuse_layer,
            name="user_tower",
        )

    def compute_item_base_embeddings(
        self, item_indices, sparse_indices=None, dense_values=None, reuse_layer=False
    ):
        item_embed = tf.nn.embedding_lookup(self.item_embeds_var, item_indices)
        concat_embeds = [item_embed]
        if sparse_indices is not None:
            sparse_embed = compute_sparse_feats(
                self.data_info,
                self.multi_sparse_combiner,
                sparse_indices,
                var_name="sparse_embeds_var",
                var_shape=(self.sparse_feature_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                reuse_layer=reuse_layer,
                flatten=True,
            )
            concat_embeds.append(sparse_embed)
        if dense_values is not None:
            dense_embed = self._compute_dense_feats(
                dense_values,
                self.data_info.item_dense_col.index,
            )
            concat_embeds.append(dense_embed)
        item_inputs = (
            tf.concat(concat_embeds, axis=1)
            if len(concat_embeds) > 1
            else concat_embeds[0]
        )
        return dense_nn(
            item_inputs,
            [*self.hidden_units, self.embed_size],
            activation=tf.nn.leaky_relu,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            reuse_layer=reuse_layer,
            name="item_tower",
        )

    def build_full_user_embeddings(self):
        user_indices = tf.range(self.n_users, dtype=tf.int32)
        sparse_indices = None
        dense_values = None
        if self.user_sparse:
            sparse_indices = tf.constant(
                self.data_info.user_sparse_unique[:-1], dtype=tf.int32
            )
        if self.user_dense:
            dense_values = tf.constant(
                self.data_info.user_dense_unique[:-1], dtype=tf.float32
            )
        return self.compute_user_base_embeddings(
            user_indices, sparse_indices, dense_values, reuse_layer=True
        )

    def build_full_item_embeddings(self):
        item_indices = tf.range(self.n_items, dtype=tf.int32)
        sparse_indices = None
        dense_values = None
        if self.has_item_sparse_feats:
            sparse_indices = tf.constant(
                self.data_info.item_sparse_unique[:-1], dtype=tf.int32
            )
        if self.has_item_dense_feats:
            dense_values = tf.constant(
                self.data_info.item_dense_unique[:-1], dtype=tf.float32
            )
        return self.compute_item_base_embeddings(
            item_indices, sparse_indices, dense_values, reuse_layer=True
        )

    def pool_sequence_embeddings(self, final_item_embeds):
        oov_embed = tf.zeros((1, self.embedding_dim), dtype=tf.float32)
        item_embeds = tf.concat([final_item_embeds, oov_embed], axis=0)
        seq_embeds = tf.nn.embedding_lookup(item_embeds, self.user_interacted_seq)
        seq_mask = tf.sequence_mask(
            self.user_interacted_len, self.max_seq_len, dtype=tf.float32
        )
        seq_mask = tf.expand_dims(seq_mask, axis=-1)
        seq_sum = tf.reduce_sum(seq_embeds * seq_mask, axis=1)
        seq_len = tf.maximum(tf.reduce_sum(seq_mask, axis=1), 1.0)
        return tf.math.divide_no_nan(seq_sum, seq_len)

    def build_query_tower(self, final_item_embeds):
        user_base = self.compute_user_base_embeddings(
            self.user_indices,
            self.user_sparse_indices if self.user_sparse else None,
            self.user_dense_values if self.user_dense else None,
            reuse_layer=True,
        )
        seq_embed = self.pool_sequence_embeddings(final_item_embeds)
        query_input = tf.concat([user_base, seq_embed], axis=1)
        query_embeds = dense_nn(
            query_input,
            self.query_hidden_units,
            activation=tf.nn.leaky_relu,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            reuse_layer=True,
            name="query_tower",
        )
        return (
            normalize_embeds(query_embeds, backend="tf")
            if self.norm_embed
            else query_embeds
        )

    def build_sparse_graph(self, indices, values, shape):
        return build_tf_sparse_tensor(indices, values, shape)

    def _compute_dense_feats(self, dense_values, dense_col_indices):
        batch_size = tf.shape(dense_values)[0]
        dense_embed = tf.gather(
            self.dense_embeds_var, dense_col_indices, axis=0
        )
        dense_embed = tf.expand_dims(dense_embed, axis=0)
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        return tf.keras.layers.Flatten()(dense_values[:, :, tf.newaxis] * dense_embed)

    def adjust_logits(self, logits, all_adjust=True):
        if self.use_correction and all_adjust:
            correction = tf.clip_by_value(self.correction, 1e-8, 1.0)
            logits -= tf.reshape(tf.math.log(correction), (1, -1))
        return logits

    def fit(
        self,
        train_data,
        neg_sampling,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        num_workers=0,
    ):
        if self.loss_type == "softmax" and self.use_correction:
            self.item_corrections = np.ones(self.n_items, dtype=np.float32)
            indices, item_counts = np.unique(
                train_data.item_indices, return_counts=True
            )
            self.item_corrections[indices] = item_counts / len(train_data)
        super().fit(
            train_data,
            neg_sampling,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
            num_workers,
        )

    def dyn_user_embedding(
        self,
        user,
        user_feats=None,
        seq=None,
        include_bias=False,
        inner_id=False,
    ):
        from ..recommendation import check_dynamic_rec_feats
        from ..recommendation.preprocess import process_embed_feat, process_embed_seq

        check_dynamic_rec_feats(self.model_name, user, user_feats, seq)
        if user is None:
            user_id = None
            user_indices = np.arange(self.n_users, dtype=np.int32)
        else:
            user_id = user_indices = self.convert_array_id(user, inner_id)
        sparse_indices, dense_values = process_embed_feat(
            self.data_info, user_id, user_feats
        )
        seq, seq_len = process_embed_seq(self, user_id, seq, inner_id)
        feed_dict = get_feed_dict(
            model=self,
            user_indices=user_indices,
            user_sparse_indices=sparse_indices,
            user_dense_values=dense_values,
            user_interacted_seq=seq,
            user_interacted_len=seq_len,
            is_training=False,
        )
        user_embeds = self.sess.run(self.serving_user_embeds, feed_dict)
        if include_bias and getattr(self, "item_biases", None) is not None:
            user_biases = np.ones([len(user_embeds), 1], dtype=user_embeds.dtype)
            user_embeds = np.hstack([user_embeds, user_biases])
        return user_embeds if user_id is None else np.squeeze(user_embeds, axis=0)

    def set_embeddings(self):
        self._assign_user_oov(var_name="user_embeds_var", scope_name="embedding")
        self.user_embeds_np = self.dyn_user_embedding(user=None, include_bias=True)
        self.item_embeds_np = self.sess.run(self.final_item_embeds)

    def build_topk(self):
        self.k = tf.placeholder(tf.int32, shape=(), name="k")
        user_embeds = tf.squeeze(self.serving_user_embeds, axis=0)
        item_embeds = self.final_item_embeds
        if self.norm_embed:
            user_embeds = normalize_embeds(user_embeds[tf.newaxis, :], backend="tf")
            item_embeds = normalize_embeds(item_embeds, backend="tf")
            user_embeds = tf.squeeze(user_embeds, axis=0)
        preds = tf.linalg.matvec(item_embeds, user_embeds)
        _, indices = tf.math.top_k(preds, self.k, sorted=True)
        return indices

    @abc.abstractmethod
    def refresh_graph(self):
        raise NotImplementedError

    def rebuild_model(self, path, model_name, full_assign=True):
        self.refresh_graph()
        rebuild_tf_model(self, path, model_name, full_assign)
