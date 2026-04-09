"""Implementation of NGCF."""
from ..bases import GraphEmbedBase
from ..graph import merge_node_embeddings, split_node_embeddings
from ..tfops import dropout_config, reg_config, tf
from ..utils.misc import count_params, hidden_units_config


class NGCF(GraphEmbedBase):
    """*Neural Graph Collaborative Filtering* algorithm.

    ``NGCF`` propagates user and item embeddings on the user-item interaction
    graph with learnable message passing. Each layer mixes neighborhood messages
    through a linear sum term and a bi-interaction term, applies a nonlinearity,
    and concatenates normalized outputs from every layer into the final
    embedding.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. ``NGCF`` only supports ranking.
    data_info : :class:`~recora.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'ranknet', 'bpr', 'lambdarank', 'listnet', 'approx_ndcg'}, default: 'bpr'
        Loss for model training.
    embed_size : int, default: 16
        Vector size of the initial user and item embeddings.
    layer_sizes : int, list of int or tuple of (int,), default: (16, 16, 16)
        Output size of each graph propagation layer. The final embedding
        dimension is ``embed_size + sum(layer_sizes)`` because all normalized
        layer outputs are concatenated.
    node_dropout_rate : float or None, default: None
        Dropout rate applied to the sparse interaction graph during message
        passing. If it is None, node dropout is disabled.
    message_dropout_rate : float or None, default: None
        Dropout rate applied to layer outputs after nonlinear message passing.
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
    ``node_dropout_rate`` and ``message_dropout_rate`` control different parts
    of the model: node dropout sparsifies the interaction graph, while message
    dropout regularizes dense hidden representations after aggregation.

    References
    ----------
    [1] *Xiang Wang et al.* `Neural Graph Collaborative Filtering
    <https://arxiv.org/abs/1905.08108>`_.
    """

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="bpr",
        embed_size=16,
        layer_sizes=(16, 16, 16),
        node_dropout_rate=None,
        message_dropout_rate=None,
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
        layer_sizes = hidden_units_config(layer_sizes)
        if any(size <= 0 for size in layer_sizes):
            raise ValueError(f"`layer_sizes` must be positive, got `{layer_sizes}`")

        super().__init__(
            task=task,
            data_info=data_info,
            embed_size=embed_size,
            embedding_dim=embed_size + sum(layer_sizes),
            tf_sess_config=tf_sess_config,
        )

        if task != "ranking":
            raise ValueError("`NGCF` is only suitable for ranking")
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

        self.all_args = locals()
        self.loss_type = loss_type
        self.listnet_temperature = listnet_temperature
        self.approx_ndcg_temperature = approx_ndcg_temperature
        self.listwise_num_pos = listwise_num_pos
        self.layer_sizes = layer_sizes
        self.node_dropout_rate = dropout_config(node_dropout_rate)
        self.message_dropout_rate = dropout_config(message_dropout_rate)
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

        interaction_graph = self.build_graph_tensor(self.node_dropout_rate)
        node_embeds = merge_node_embeddings(user_embeds_var, item_embeds_var)
        current_embeds = node_embeds
        all_node_embeds = [tf.nn.l2_normalize(node_embeds, axis=1)]
        input_dim = self.embed_size

        for layer_idx, output_dim in enumerate(self.layer_sizes):
            side_embeds = tf.sparse_tensor_dense_matmul(interaction_graph, current_embeds)
            with tf.variable_scope(f"ngcf_layer_{layer_idx}"):
                sum_weights = tf.get_variable(
                    name="sum_weights",
                    shape=(input_dim, output_dim),
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
                sum_bias = tf.get_variable(
                    name="sum_bias",
                    shape=(output_dim,),
                    initializer=tf.zeros_initializer(),
                    regularizer=self.reg,
                )
                bi_weights = tf.get_variable(
                    name="bi_weights",
                    shape=(input_dim, output_dim),
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
                bi_bias = tf.get_variable(
                    name="bi_bias",
                    shape=(output_dim,),
                    initializer=tf.zeros_initializer(),
                    regularizer=self.reg,
                )

            sum_messages = tf.matmul(side_embeds, sum_weights) + sum_bias
            bi_messages = (
                tf.matmul(current_embeds * side_embeds, bi_weights) + bi_bias
            )
            current_embeds = tf.nn.leaky_relu(sum_messages + bi_messages, alpha=0.2)
            current_embeds = self._apply_message_dropout(current_embeds, layer_idx)
            all_node_embeds.append(tf.nn.l2_normalize(current_embeds, axis=1))
            input_dim = output_dim

        final_node_embeds = tf.concat(all_node_embeds, axis=1)
        final_user_embeds, final_item_embeds = split_node_embeddings(
            final_node_embeds, self.n_users, self.n_items
        )
        self.set_output_embeddings(final_user_embeds, final_item_embeds)
        count_params()

    def _apply_message_dropout(self, inputs, layer_idx):
        if not self.message_dropout_rate:
            return inputs
        keep_prob = 1.0 - self.message_dropout_rate
        return tf.cond(
            self.is_training,
            lambda: tf.nn.dropout(
                inputs, keep_prob=keep_prob, seed=self.seed + layer_idx
            ),
            lambda: inputs,
        )
