import numpy as np

from ..tfops import tf


def build_bipartite_adjacency(
    user_consumed,
    n_users,
    n_items,
    normalization="symmetric",
    add_self_loops=False,
):
    num_nodes = n_users + n_items
    row_indices = []
    col_indices = []

    for user, items in user_consumed.items():
        for item in items:
            item_node = n_users + item
            row_indices.extend((user, item_node))
            col_indices.extend((item_node, user))

    if add_self_loops:
        self_loops = np.arange(num_nodes, dtype=np.int64)
        row_indices.extend(self_loops.tolist())
        col_indices.extend(self_loops.tolist())

    indices = np.column_stack([row_indices, col_indices]).astype(np.int64)
    if len(indices) == 0:
        raise ValueError("interaction graph is empty")

    degrees = np.bincount(indices[:, 0], minlength=num_nodes).astype(np.float32)
    values = np.ones(len(indices), dtype=np.float32)

    if normalization == "symmetric":
        degree_scale = np.zeros_like(degrees)
        non_zero_mask = degrees > 0
        degree_scale[non_zero_mask] = np.power(degrees[non_zero_mask], -0.5)
        values = degree_scale[indices[:, 0]] * degree_scale[indices[:, 1]]
    elif normalization == "left":
        degree_scale = np.zeros_like(degrees)
        non_zero_mask = degrees > 0
        degree_scale[non_zero_mask] = 1.0 / degrees[non_zero_mask]
        values = degree_scale[indices[:, 0]]
    elif normalization in (None, "none"):
        pass
    else:
        raise ValueError(f"unsupported adjacency normalization: {normalization}")

    order = np.lexsort((indices[:, 1], indices[:, 0]))
    indices = indices[order]
    values = values[order]
    shape = np.array([num_nodes, num_nodes], dtype=np.int64)
    return indices, values, shape


def build_tf_sparse_tensor(indices, values, shape):
    sparse_tensor = tf.SparseTensor(
        indices=tf.constant(indices, dtype=tf.int64),
        values=tf.constant(values, dtype=tf.float32),
        dense_shape=tf.constant(shape, dtype=tf.int64),
    )
    return tf.sparse_reorder(sparse_tensor)


def sparse_dropout(sparse_tensor, keep_prob, seed):
    if keep_prob <= 0.0 or keep_prob > 1.0:
        raise ValueError(f"`keep_prob` must be in (0.0, 1.0], got {keep_prob}")
    if keep_prob == 1.0:
        return tf.sparse_reorder(sparse_tensor)

    random_tensor = keep_prob + tf.random_uniform(
        tf.shape(sparse_tensor.values), seed=seed
    )
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    dropped = tf.SparseTensor(
        indices=tf.boolean_mask(sparse_tensor.indices, dropout_mask),
        values=tf.boolean_mask(sparse_tensor.values, dropout_mask)
        / tf.constant(keep_prob, dtype=tf.float32),
        dense_shape=sparse_tensor.dense_shape,
    )
    return tf.sparse_reorder(dropped)


def merge_node_embeddings(user_embeds, item_embeds):
    return tf.concat([user_embeds, item_embeds], axis=0)


def split_node_embeddings(node_embeds, n_users, n_items):
    return node_embeds[:n_users], node_embeds[n_users : n_users + n_items]
