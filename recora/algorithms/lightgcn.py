"""Implementation of LightGCN."""
from ..bases import GraphEmbedBase
from ..graph import merge_node_embeddings, split_node_embeddings
from ..tfops import reg_config, tf
from ..utils.misc import count_params


class LightGCN(GraphEmbedBase):
    """*LightGCN* algorithm.

    ``LightGCN`` is a graph collaborative filtering model for implicit ranking.
    It propagates user and item embeddings on the normalized user-item
    interaction graph and averages the embeddings from all propagation layers,
    including the input layer. Unlike earlier graph recommenders, it keeps the
    propagation step intentionally simple and does not apply feature transforms
    or nonlinear activations inside graph convolution.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. ``LightGCN`` only supports ranking.
    data_info : :class:`~recora.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'ranknet', 'bpr', 'lambdarank', 'listnet', 'approx_ndcg'}, default: 'bpr'
        Loss for model training.
    embed_size : int, default: 16
        Vector size of user and item embeddings.
    n_layers : int, default: 3
        Number of graph propagation layers. The final embedding is the mean of
        layer-0 embeddings and the outputs from all ``n_layers`` propagations.
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
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative sampling strategy used by sampled losses.
    num_neg : int, default: 1
        Number of negative samples for each positive sample.
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
    The interaction graph comes from the bipartite user-item matrix and is
    normalized by :class:`~recora.bases.GraphEmbedBase`. Training is still done
    on user-item examples; the graph propagation only affects how embeddings are
    generated before scoring.

    References
    ----------
    [1] *Xiangnan He et al.* `LightGCN: Simplifying and Powering Graph
    Convolution Network for Recommendation
    <https://arxiv.org/abs/2002.02126>`_.
    """

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="bpr",
        embed_size=16,
        n_layers=3,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
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
            embedding_dim=embed_size,
            tf_sess_config=tf_sess_config,
        )

        if task != "ranking":
            raise ValueError("`LightGCN` is only suitable for ranking")
        if loss_type not in (
            "cross_entropy",
            "focal",
            "ranknet",
            "bpr",
            "lambdarank",
            "listnet",
            "approx_ndcg",
        ):
            raise ValueError(f"unsupported `loss_type`: `{loss_type}`")
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError(f"`n_layers` must be a positive integer, got `{n_layers}`")

        self.all_args = locals()
        self.loss_type = loss_type
        self.listnet_temperature = listnet_temperature
        self.approx_ndcg_temperature = approx_ndcg_temperature
        self.listwise_num_pos = listwise_num_pos
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.seed = seed

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.build_placeholders()

        with tf.variable_scope("embedding"):
            user_embeds_var = tf.get_variable(
                name="user_embeds_var",
                shape=(self.n_users, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            item_embeds_var = tf.get_variable(
                name="item_embeds_var",
                shape=(self.n_items, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )

        interaction_graph = self.build_graph_tensor()
        node_embeds = merge_node_embeddings(user_embeds_var, item_embeds_var)
        all_node_embeds = [node_embeds]
        propagated_embeds = node_embeds
        for _ in range(self.n_layers):
            propagated_embeds = tf.sparse_tensor_dense_matmul(
                interaction_graph, propagated_embeds
            )
            all_node_embeds.append(propagated_embeds)

        final_node_embeds = tf.add_n(all_node_embeds) / float(len(all_node_embeds))
        final_user_embeds, final_item_embeds = split_node_embeddings(
            final_node_embeds, self.n_users, self.n_items
        )
        self.set_output_embeddings(final_user_embeds, final_item_embeds)
        count_params()
