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


def build_bipartite_neighbors(user_consumed, item_consumed, n_users, n_items):
    neighbors = []
    for user in range(n_users):
        consumed_items = np.asarray(sorted(user_consumed.get(user, [])), dtype=np.int32)
        neighbors.append(consumed_items + np.int32(n_users))
    for item in range(n_items):
        consuming_users = np.asarray(sorted(item_consumed.get(item, [])), dtype=np.int32)
        neighbors.append(consuming_users)
    return neighbors


def sample_bipartite_neighbors(
    neighbors,
    n_users,
    user_roots,
    item_roots,
    sample_sizes,
    rng,
):
    user_roots = np.asarray(user_roots, dtype=np.int32).reshape(-1)
    item_roots = np.asarray(item_roots, dtype=np.int32).reshape(-1)
    root_nodes = []
    if len(user_roots) > 0:
        root_nodes.extend(user_roots.tolist())
    if len(item_roots) > 0:
        root_nodes.extend((item_roots + np.int32(n_users)).tolist())
    if not root_nodes:
        raise ValueError("sampled GraphSAGE subgraph requires at least one root node")

    sampled_nodes = set(root_nodes)
    sampled_edges = []
    frontier = np.asarray(sorted(set(root_nodes)), dtype=np.int32)

    for fanout in sample_sizes:
        next_frontier = set()
        for node in frontier:
            node_neighbors = neighbors[node]
            if len(node_neighbors) == 0:
                continue
            if fanout >= len(node_neighbors):
                sampled_neighbors = node_neighbors
            else:
                sampled_neighbors = np.sort(
                    rng.choice(node_neighbors, size=fanout, replace=False).astype(
                        np.int32
                    )
                )
            for neighbor in sampled_neighbors:
                neighbor = int(neighbor)
                sampled_edges.append((int(node), neighbor))
                sampled_nodes.add(neighbor)
                next_frontier.add(neighbor)
        frontier = np.asarray(sorted(next_frontier), dtype=np.int32)

    sampled_user_nodes = np.asarray(
        sorted(node for node in sampled_nodes if node < n_users), dtype=np.int32
    )
    sampled_item_nodes = np.asarray(
        sorted(node - n_users for node in sampled_nodes if node >= n_users),
        dtype=np.int32,
    )
    user_positions = {node: idx for idx, node in enumerate(sampled_user_nodes)}
    item_positions = {node: idx for idx, node in enumerate(sampled_item_nodes)}
    user_offset = len(sampled_user_nodes)

    if sampled_edges:
        local_indices = np.asarray(
            [
                (
                    user_positions[row]
                    if row < n_users
                    else user_offset + item_positions[row - n_users],
                    user_positions[col]
                    if col < n_users
                    else user_offset + item_positions[col - n_users],
                )
                for row, col in sampled_edges
            ],
            dtype=np.int64,
        )
        row_degrees = np.bincount(
            local_indices[:, 0], minlength=user_offset + len(sampled_item_nodes)
        ).astype(np.float32)
        values = np.divide(
            1.0,
            row_degrees[local_indices[:, 0]],
            out=np.zeros(len(local_indices), dtype=np.float32),
            where=row_degrees[local_indices[:, 0]] > 0,
        )
        order = np.lexsort((local_indices[:, 1], local_indices[:, 0]))
        indices = local_indices[order]
        values = values[order]
        node_has_neighbors = row_degrees > 0
    else:
        indices = np.zeros((0, 2), dtype=np.int64)
        values = np.zeros(0, dtype=np.float32)
        node_has_neighbors = np.zeros(
            len(sampled_user_nodes) + len(sampled_item_nodes), dtype=bool
        )

    user_root_positions = np.asarray(
        [user_positions[user] for user in user_roots], dtype=np.int32
    )
    item_root_positions = np.asarray(
        [item_positions[item] for item in item_roots], dtype=np.int32
    )

    return (
        indices,
        values,
        sampled_user_nodes,
        sampled_item_nodes,
        node_has_neighbors,
        user_root_positions,
        item_root_positions,
    )


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


def build_weighted_item_adjacency(
    user_consumed,
    item_consumed,
    n_items,
    neighbor_topk,
    num_walks,
    walk_length,
    restart_prob,
    seed,
):
    rng = np.random.default_rng(seed)
    row_indices = []
    col_indices = []
    values = []
    has_neighbors = []

    for item in range(n_items):
        weights = _random_walk_item_neighbors(
            user_consumed,
            item_consumed,
            item,
            num_walks,
            walk_length,
            restart_prob,
            rng,
        )
        if not weights:
            has_neighbors.append(False)
            continue

        sorted_neighbors = sorted(weights.items(), key=lambda x: (-x[1], x[0]))
        sorted_neighbors = sorted_neighbors[:neighbor_topk]
        total_weight = float(sum(weight for _, weight in sorted_neighbors))
        has_neighbors.append(total_weight > 0.0)
        if total_weight <= 0.0:
            continue

        for neighbor, weight in sorted_neighbors:
            row_indices.append(item)
            col_indices.append(neighbor)
            values.append(weight / total_weight)

    if row_indices:
        indices = np.column_stack([row_indices, col_indices]).astype(np.int64)
        values = np.asarray(values, dtype=np.float32)
        order = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[order]
        values = values[order]
    else:
        indices = np.zeros((0, 2), dtype=np.int64)
        values = np.zeros(0, dtype=np.float32)
    shape = np.array([n_items, n_items], dtype=np.int64)
    return indices, values, shape, has_neighbors


def _random_walk_item_neighbors(
    user_consumed,
    item_consumed,
    src_item,
    num_walks,
    walk_length,
    restart_prob,
    rng,
):
    weights = dict()
    users = item_consumed.get(src_item, [])
    if not users:
        return weights

    for _ in range(num_walks):
        current_item = src_item
        for _ in range(walk_length):
            if rng.random() < restart_prob:
                current_item = src_item
            connected_users = item_consumed.get(current_item, [])
            if not connected_users:
                break
            sampled_user = int(rng.choice(connected_users))
            consumed_items = user_consumed.get(sampled_user, [])
            if not consumed_items:
                break
            next_item = int(rng.choice(consumed_items))
            if next_item != src_item:
                weights[next_item] = weights.get(next_item, 0.0) + 1.0
            current_item = next_item
    return weights
