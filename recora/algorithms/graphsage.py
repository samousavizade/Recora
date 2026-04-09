"""Implementation of GraphSage."""

import numpy as np

from ..bases import GraphFeatBase
from ..graph import (
    build_bipartite_adjacency,
    build_bipartite_neighbors,
    merge_node_embeddings,
    split_node_embeddings,
)
from ..layers import normalize_embeds, shared_dense
from ..tfops import tf
from ..utils.misc import count_params


class GraphSage(GraphFeatBase):
    """*GraphSAGE* graph feature retrieval model.

    ``GraphSage`` in Recora combines feature-aware user/item towers with graph
    propagation on the user-item interaction graph. Item embeddings are enriched
    through GraphSAGE-style neighborhood aggregation, while user retrieval can
    still leverage user features and recent interaction sequences through the
    dynamic query tower inherited from :class:`~recora.bases.GraphFeatBase`.

    This implementation supports two graph training modes:

    - Full-graph propagation, where every layer aggregates over the complete
      bipartite interaction graph.
    - Paper-style fixed-fanout neighborhood sampling during training via
      ``neighbor_sampling=True`` and ``sample_sizes``.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. ``GraphSage`` only supports ranking.
    data_info : :class:`~recora.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'max_margin', 'softmax', 'listnet', 'approx_ndcg'}, default: 'softmax'
        Loss for model training.
    embed_size : int, default: 16
        Vector size of the base user and item embeddings before graph
        propagation.
    hidden_units : int, list of int or tuple of (int,), default: (128, 64)
        Hidden layer sizes for the feature towers used to build base user/item
        embeddings and the dynamic query tower.
    layer_sizes : int, list of int or tuple of (int,), default: (64, 64)
        Output size of each GraphSAGE propagation layer. The number of layers
        also determines the number of hops used when ``neighbor_sampling`` is
        enabled.
    norm_embed : bool, default: False
        Whether to l2 normalize query embeddings before inference.
    n_epochs : int, default: 20
        Number of epochs for training.
    lr : float, default: 0.001
        Learning rate for training.
    lr_decay : bool, default: False
        Whether to use learning rate decay.
    epsilon : float, default: 1e-5
        A small constant added to the denominator to improve numerical
        stability in Adam optimizer.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    batch_size : int, default: 256
        Batch size for training.
    num_neg : int, default: 1
        Number of negative samples for each positive sample.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative item sampling strategy used by sampled losses. This parameter
        does not control GraphSAGE neighborhood sampling.
    use_bn : bool, default: True
        Whether to use batch normalization in the feature towers.
    dropout_rate : float or None, default: None
        Dropout rate applied in the feature towers. If it is None, dropout is
        disabled.
    recent_num : int or None, default: 10
        Number of most recent interacted items kept in the dynamic user
        sequence. If specified, ``random_num`` is ignored.
    random_num : int or None, default: None
        Number of randomly sampled interacted items kept in the dynamic user
        sequence when ``recent_num`` is None.
    use_correction : bool, default: True
        Whether to use sampling-bias correction in softmax loss.
    multi_sparse_combiner : {'sum', 'mean', 'sqrtn'}, default: 'sqrtn'
        Combiner used for multi-sparse features.
    neighbor_sampling : bool, default: False
        Whether to use fixed-fanout neighborhood sampling during training. When
        disabled, training uses the full graph at every layer.
    sample_sizes : tuple of int or None, default: None
        Fanout for each sampled GraphSAGE hop when ``neighbor_sampling`` is
        enabled. The tuple length must match ``layer_sizes``. If omitted,
        ``(25,) * len(layer_sizes)`` is used.
    seed : int, default: 42
        Random seed.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.
    listnet_temperature : float, default: 1.0
        Temperature used in ``listnet`` loss.
    approx_ndcg_temperature : float, default: 1.0
        Temperature used in ``approx_ndcg`` loss.

    Notes
    -----
    ``neighbor_sampling`` only changes the training-time graph encoder. Inference,
    recommendation, save/load, and retraining still materialize full-graph user
    and item embeddings for deterministic serving behavior.

    References
    ----------
    [1] *Will Hamilton et al.* `Inductive Representation Learning on Large
    Graphs <https://arxiv.org/abs/1706.02216>`_.
    """

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="softmax",
        embed_size=16,
        hidden_units=(128, 64),
        layer_sizes=(64, 64),
        norm_embed=False,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        num_neg=1,
        sampler="random",
        use_bn=True,
        dropout_rate=None,
        recent_num=10,
        random_num=None,
        use_correction=True,
        multi_sparse_combiner="sqrtn",
        neighbor_sampling=False,
        sample_sizes=None,
        seed=42,
        tf_sess_config=None,
        listnet_temperature=1.0,
        approx_ndcg_temperature=1.0,
        listwise_num_pos=1,
):
        super().__init__(
            task=task,
            data_info=data_info,
            embed_size=embed_size,
            norm_embed=norm_embed,
            hidden_units=hidden_units,
            layer_sizes=layer_sizes,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            reg=reg,
            recent_num=recent_num,
            random_num=random_num,
            multi_sparse_combiner=multi_sparse_combiner,
            tf_sess_config=tf_sess_config,
            use_correction=use_correction,
        )
        if task != "ranking":
            raise ValueError("`GraphSage` is only suitable for ranking")
        if loss_type not in (
            "cross_entropy",
            "max_margin",
            "softmax",
            "listnet",
            "approx_ndcg",
        ):
            raise ValueError(f"Unsupported `loss_type`: `{loss_type}`")
        if not isinstance(neighbor_sampling, bool):
            raise ValueError("`neighbor_sampling` must be bool")
        sample_sizes = self._resolve_sample_sizes(sample_sizes, neighbor_sampling)
        self.all_args = locals()
        self.loss_type = loss_type
        self.listnet_temperature = listnet_temperature
        self.approx_ndcg_temperature = approx_ndcg_temperature
        self.listwise_num_pos = listwise_num_pos
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.sampler = sampler
        self.neighbor_sampling = neighbor_sampling
        self.sample_sizes = sample_sizes
        self.seed = seed
        self.refresh_graph()

    def _resolve_sample_sizes(self, sample_sizes, neighbor_sampling):
        if sample_sizes is None:
            return tuple([25] * len(self.layer_sizes)) if neighbor_sampling else None
        sample_sizes = tuple(sample_sizes)
        if len(sample_sizes) != len(self.layer_sizes):
            raise ValueError("`sample_sizes` must match `layer_sizes` in length")
        if any(not isinstance(size, int) or size <= 0 for size in sample_sizes):
            raise ValueError("`sample_sizes` must be a tuple of positive integers")
        return sample_sizes

    def build_placeholders(self):
        super().build_placeholders()
        if self.neighbor_sampling:
            self.sampled_graph_indices = tf.placeholder(
                tf.int64, shape=[None, 2], name="sampled_graph_indices"
            )
            self.sampled_graph_values = tf.placeholder(
                tf.float32, shape=[None], name="sampled_graph_values"
            )
            self.sampled_user_nodes = tf.placeholder(
                tf.int32, shape=[None], name="sampled_user_nodes"
            )
            self.sampled_item_nodes = tf.placeholder(
                tf.int32, shape=[None], name="sampled_item_nodes"
            )
            self.sampled_node_has_neighbors = tf.placeholder(
                tf.bool, shape=[None], name="sampled_node_has_neighbors"
            )
            self.sampled_batch_user_positions = tf.placeholder(
                tf.int32, shape=[None], name="sampled_batch_user_positions"
            )
            self.sampled_batch_item_positions = tf.placeholder(
                tf.int32, shape=[None], name="sampled_batch_item_positions"
            )
            if self.loss_type == "max_margin":
                self.sampled_batch_item_neg_positions = tf.placeholder(
                    tf.int32, shape=[None], name="sampled_batch_item_neg_positions"
                )

    def refresh_graph(self):
        (
            self.graph_indices,
            self.graph_values,
            self.graph_shape,
        ) = build_bipartite_adjacency(
            user_consumed=self.user_consumed,
            n_users=self.n_users,
            n_items=self.n_items,
            normalization="left",
            add_self_loops=False,
        )
        self.graph_neighbors = build_bipartite_neighbors(
            user_consumed=self.user_consumed,
            item_consumed=self.data_info.item_consumed,
            n_users=self.n_users,
            n_items=self.n_items,
        )
        self.node_has_neighbors = [len(neighbors) > 0 for neighbors in self.graph_neighbors]

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.build_placeholders()
        self.build_variables()
        self.final_user_embeds, self.final_item_embeds = self._build_full_graph_embeddings()
        self.final_user_graph_embeds = self.final_user_embeds
        self.graph_user_embeds = tf.nn.embedding_lookup(self.final_user_embeds, self.user_indices)
        self.serving_user_embeds = self.build_query_tower(self.final_item_embeds)
        self.lookup_item_embeds = tf.nn.embedding_lookup(
            self.final_item_embeds, self.item_indices
        )
        if self.neighbor_sampling:
            self.user_embeds, self.item_embeds = self._build_sampled_training_embeddings()
        else:
            self.user_embeds = self.graph_user_embeds
            self.item_embeds = self.lookup_item_embeds
        if self.loss_type in ("cross_entropy", "listnet", "approx_ndcg"):
            self.output = tf.reduce_sum(self.user_embeds * self.item_embeds, axis=1)
        elif self.loss_type == "max_margin":
            if self.neighbor_sampling:
                self.item_embeds_neg = tf.gather(
                    self.sampled_item_graph_embeds,
                    self.sampled_batch_item_neg_positions,
                )
            else:
                self.item_embeds_neg = tf.nn.embedding_lookup(
                    self.final_item_embeds, self.item_indices_neg
                )
            if self.norm_embed:
                self.item_embeds_neg = normalize_embeds(
                    self.item_embeds_neg, backend="tf"
                )
        self.serving_topk = self.build_topk()
        count_params()

    def _build_full_graph_embeddings(self):
        user_base_embeds = self.build_full_user_embeddings()
        item_base_embeds = self.build_full_item_embeddings()
        graph = self.build_sparse_graph(
            self.graph_indices, self.graph_values, self.graph_shape
        )
        node_has_neighbors = tf.constant(self.node_has_neighbors, dtype=tf.bool)
        return self._propagate_graph_embeddings(
            user_base_embeds, item_base_embeds, graph, node_has_neighbors
        )

    def _build_sampled_training_embeddings(self):
        user_base_embeds = self.build_user_embeddings_from_indices(self.sampled_user_nodes)
        item_base_embeds = self.build_item_embeddings_from_indices(self.sampled_item_nodes)
        num_nodes = tf.shape(self.sampled_user_nodes)[0] + tf.shape(self.sampled_item_nodes)[0]
        graph = tf.SparseTensor(
            indices=self.sampled_graph_indices,
            values=self.sampled_graph_values,
            dense_shape=tf.cast(tf.stack([num_nodes, num_nodes]), dtype=tf.int64),
        )
        graph = tf.sparse_reorder(graph)
        sampled_user_embeds, sampled_item_embeds = self._propagate_graph_embeddings(
            user_base_embeds,
            item_base_embeds,
            graph,
            self.sampled_node_has_neighbors,
        )
        self.sampled_user_graph_embeds = sampled_user_embeds
        self.sampled_item_graph_embeds = sampled_item_embeds
        user_embeds = tf.gather(sampled_user_embeds, self.sampled_batch_user_positions)
        item_embeds = tf.gather(sampled_item_embeds, self.sampled_batch_item_positions)
        return user_embeds, item_embeds

    def _propagate_graph_embeddings(
        self, user_base_embeds, item_base_embeds, graph, node_has_neighbors
    ):
        node_embeds = merge_node_embeddings(user_base_embeds, item_base_embeds)
        node_mask = tf.cast(node_has_neighbors, dtype=tf.bool)[:, None]
        for layer_idx, layer_size in enumerate(self.layer_sizes, start=1):
            agg_embeds = tf.sparse_tensor_dense_matmul(graph, node_embeds)
            broadcast_mask = tf.tile(node_mask, [1, tf.shape(agg_embeds)[1]])
            agg_embeds = tf.where(broadcast_mask, agg_embeds, node_embeds)
            layer_inputs = tf.concat([node_embeds, agg_embeds], axis=1)
            node_embeds = shared_dense(
                layer_inputs,
                layer_size,
                activation=tf.nn.leaky_relu,
                name=f"graphsage_layer_{layer_idx}",
                scope_name="graphsage",
            )
            node_embeds = normalize_embeds(node_embeds, backend="tf")
        return tf.split(
            node_embeds,
            [tf.shape(user_base_embeds)[0], tf.shape(item_base_embeds)[0]],
            axis=0,
        )

    def dyn_user_embedding(
        self,
        user,
        user_feats=None,
        seq=None,
        include_bias=False,
        inner_id=False,
    ):
        if user_feats is None and seq is None:
            if user is None:
                user_embeds = self.sess.run(self.final_user_embeds)
                if include_bias and getattr(self, "item_biases", None) is not None:
                    user_biases = np.ones([len(user_embeds), 1], dtype=user_embeds.dtype)
                    user_embeds = np.hstack([user_embeds, user_biases])
                return user_embeds

            user_indices = self.convert_array_id(user, inner_id)
            if np.all(user_indices < self.n_users):
                feed_dict = {self.user_indices: user_indices, self.is_training: False}
                user_embeds = self.sess.run(self.graph_user_embeds, feed_dict)
                if include_bias and getattr(self, "item_biases", None) is not None:
                    user_biases = np.ones([len(user_embeds), 1], dtype=user_embeds.dtype)
                    user_embeds = np.hstack([user_embeds, user_biases])
                return np.squeeze(user_embeds, axis=0)

        return super().dyn_user_embedding(
            user,
            user_feats=user_feats,
            seq=seq,
            include_bias=include_bias,
            inner_id=inner_id,
        )

    def set_embeddings(self):
        self.user_embeds_np, self.item_embeds_np = self.sess.run(
            [self.final_user_embeds, self.final_item_embeds]
        )
