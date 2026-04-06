import abc

from .embed_base import EmbedBase
from ..graph import (
    build_bipartite_adjacency,
    build_tf_sparse_tensor,
    sparse_dropout,
)
from ..tfops import rebuild_tf_model, sess_config, tf


class GraphEmbedBase(EmbedBase):
    user_variables = ("embedding/user_embeds_var",)
    item_variables = ("embedding/item_embeds_var",)

    def __init__(
        self,
        task,
        data_info,
        embed_size,
        embedding_dim=None,
        lower_upper_bound=None,
        tf_sess_config=None,
        normalization="symmetric",
        add_self_loops=False,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)
        self.sess = sess_config(tf_sess_config)
        self.embedding_dim = embedding_dim or embed_size
        self.graph_normalization = normalization
        self.graph_add_self_loops = add_self_loops
        self.graph_indices = None
        self.graph_values = None
        self.graph_shape = None
        self.final_user_embeds = None
        self.final_item_embeds = None
        self.refresh_graph()

    def refresh_graph(self):
        self.graph_indices, self.graph_values, self.graph_shape = (
            build_bipartite_adjacency(
                user_consumed=self.user_consumed,
                n_users=self.n_users,
                n_items=self.n_items,
                normalization=self.graph_normalization,
                add_self_loops=self.graph_add_self_loops,
            )
        )

    def build_graph_tensor(self, dropout_rate=0.0):
        graph = build_tf_sparse_tensor(
            self.graph_indices, self.graph_values, self.graph_shape
        )
        if not dropout_rate:
            return graph
        keep_prob = 1.0 - dropout_rate
        return tf.cond(
            self.is_training,
            lambda: sparse_dropout(graph, keep_prob=keep_prob, seed=self.seed),
            lambda: graph,
        )

    def build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

    def set_output_embeddings(self, user_embeds, item_embeds):
        self.final_user_embeds = user_embeds
        self.final_item_embeds = item_embeds
        self.user_embeds = tf.nn.embedding_lookup(user_embeds, self.user_indices)
        self.item_embeds = tf.nn.embedding_lookup(item_embeds, self.item_indices)
        self.output = tf.reduce_sum(self.user_embeds * self.item_embeds, axis=1)

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError

    def set_embeddings(self):
        self.user_embeds_np, self.item_embeds_np = self.sess.run(
            [self.final_user_embeds, self.final_item_embeds]
        )

    def rebuild_model(self, path, model_name, full_assign=True):
        self.refresh_graph()
        rebuild_tf_model(self, path, model_name, full_assign)
