"""Implementation of GraphSage."""

from ..bases import GraphFeatBase
from ..graph import (
    build_bipartite_adjacency,
    merge_node_embeddings,
    split_node_embeddings,
)
from ..layers import normalize_embeds, shared_dense
from ..tfops import tf
from ..utils.misc import count_params


class GraphSage(GraphFeatBase):
    """GraphSAGE-like graph feature retrieval model."""

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
        seed=42,
        tf_sess_config=None,
        listnet_temperature=1.0,
        approx_ndcg_temperature=1.0,
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
        self.all_args = locals()
        self.loss_type = loss_type
        self.listnet_temperature = listnet_temperature
        self.approx_ndcg_temperature = approx_ndcg_temperature
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
        num_nodes = self.n_users + self.n_items
        degrees = [0] * num_nodes
        for row_index, _ in self.graph_indices:
            degrees[row_index] += 1
        self.node_has_neighbors = [d > 0 for d in degrees]

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.build_placeholders()
        self.build_variables()
        user_base_embeds = self.build_full_user_embeddings()
        item_base_embeds = self.build_full_item_embeddings()
        graph = self.build_sparse_graph(
            self.graph_indices, self.graph_values, self.graph_shape
        )
        node_embeds = merge_node_embeddings(user_base_embeds, item_base_embeds)
        node_mask = tf.constant(self.node_has_neighbors, dtype=tf.bool)[:, None]
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

        self.final_user_graph_embeds, self.final_item_embeds = split_node_embeddings(
            node_embeds, self.n_users, self.n_items
        )
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
