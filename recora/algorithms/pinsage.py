"""Implementation of PinSage."""

from ..bases import GraphFeatBase
from ..graph import build_weighted_item_adjacency
from ..layers import normalize_embeds, shared_dense
from ..tfops import tf
from ..utils.misc import count_params


class PinSage(GraphFeatBase):
    """*PinSAGE* graph feature retrieval model.

    ``PinSage`` in Recora first constructs a weighted item-item graph from
    random walks over the user-item interaction graph, then propagates item
    embeddings on that graph. User representations are produced by the dynamic
    query tower from :class:`~recora.bases.GraphFeatBase`, so the model can use
    user features and recent interaction sequences while retrieving against
    graph-enriched item embeddings.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. ``PinSage`` only supports ranking.
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
        Output size of each item-graph propagation layer.
    neighbor_topk : int, default: 20
        Maximum number of neighbors kept for each item in the weighted
        item-item graph.
    num_walks : int, default: 10
        Number of random walks launched from each item when estimating
        neighbor importance.
    walk_length : int, default: 2
        Number of alternating item-user-item steps taken in each random walk.
    restart_prob : float, default: 0.5
        Probability of jumping back to the source item during random walks.
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
        Negative item sampling strategy used by sampled losses.
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
    The item graph is rebuilt from ``data_info`` whenever the model is created
    or rebuilt for further training. Random walks estimate item importance, and
    the final graph keeps only the top weighted neighbors for each source item.

    References
    ----------
    [1] *Rex Ying et al.* `Graph Convolutional Neural Networks for Web-Scale
    Recommender Systems <https://arxiv.org/abs/1806.01973>`_.
    """

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="softmax",
        embed_size=16,
        hidden_units=(128, 64),
        layer_sizes=(64, 64),
        neighbor_topk=20,
        num_walks=10,
        walk_length=2,
        restart_prob=0.5,
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
        seed=42,
        tf_sess_config=None,
        listnet_temperature=1.0,
        approx_ndcg_temperature=1.0,
        listwise_num_pos=1,
):
        self.neighbor_topk = neighbor_topk
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.restart_prob = restart_prob
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
            raise ValueError("`PinSage` is only suitable for ranking")
        if loss_type not in (
            "cross_entropy",
            "max_margin",
            "softmax",
            "listnet",
            "approx_ndcg",
        ):
            raise ValueError(f"Unsupported `loss_type`: `{loss_type}`")
        if neighbor_topk <= 0:
            raise ValueError("`neighbor_topk` must be a positive integer")
        if num_walks <= 0 or walk_length <= 0:
            raise ValueError("`num_walks` and `walk_length` must be positive integers")
        if not 0.0 <= restart_prob <= 1.0:
            raise ValueError("`restart_prob` must be in [0.0, 1.0]")
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
        self.seed = seed
        self.refresh_graph()

    def refresh_graph(self):
        (
            self.item_graph_indices,
            self.item_graph_values,
            self.item_graph_shape,
            self.item_has_neighbors,
        ) = build_weighted_item_adjacency(
            user_consumed=self.user_consumed,
            item_consumed=self.data_info.item_consumed,
            n_items=self.n_items,
            neighbor_topk=self.neighbor_topk,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            restart_prob=self.restart_prob,
            seed=self.seed,
        )

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.build_placeholders()
        self.build_variables()
        item_embeds = self.build_full_item_embeddings()
        graph = self.build_sparse_graph(
            self.item_graph_indices, self.item_graph_values, self.item_graph_shape
        )
        item_mask = tf.constant(self.item_has_neighbors, dtype=tf.bool)[:, None]
        for layer_idx, layer_size in enumerate(self.layer_sizes, start=1):
            agg_embeds = tf.sparse_tensor_dense_matmul(graph, item_embeds)
            broadcast_mask = tf.tile(item_mask, [1, tf.shape(agg_embeds)[1]])
            agg_embeds = tf.where(broadcast_mask, agg_embeds, item_embeds)
            layer_inputs = tf.concat([item_embeds, agg_embeds], axis=1)
            item_embeds = shared_dense(
                layer_inputs,
                layer_size,
                activation=tf.nn.leaky_relu,
                name=f"pinsage_layer_{layer_idx}",
                scope_name="pinsage",
            )
            item_embeds = normalize_embeds(item_embeds, backend="tf")

        self.final_item_embeds = item_embeds
        self.serving_user_embeds = self.build_query_tower(self.final_item_embeds)
        self.user_embeds = self.serving_user_embeds
        self.item_embeds = tf.nn.embedding_lookup(self.final_item_embeds, self.item_indices)
        if self.loss_type in ("cross_entropy", "listnet", "approx_ndcg"):
            self.output = tf.reduce_sum(self.user_embeds * self.item_embeds, axis=1)
        elif self.loss_type == "max_margin":
            self.item_embeds_neg = tf.nn.embedding_lookup(
                self.final_item_embeds, self.item_indices_neg
            )
            if self.norm_embed:
                self.item_embeds_neg = normalize_embeds(
                    self.item_embeds_neg, backend="tf"
                )
        self.serving_topk = self.build_topk()
        count_params()
