import numpy as np

from recora.graph import (
    build_bipartite_neighbors,
    build_weighted_item_adjacency,
    sample_bipartite_neighbors,
)


def test_weighted_item_adjacency_deterministic():
    user_consumed = {0: [0, 1, 2], 1: [0, 2], 2: [3]}
    item_consumed = {0: [0, 1], 1: [0], 2: [0, 1], 3: [2]}
    graph1 = build_weighted_item_adjacency(
        user_consumed,
        item_consumed,
        n_items=5,
        neighbor_topk=2,
        num_walks=5,
        walk_length=2,
        restart_prob=0.5,
        seed=7,
    )
    graph2 = build_weighted_item_adjacency(
        user_consumed,
        item_consumed,
        n_items=5,
        neighbor_topk=2,
        num_walks=5,
        walk_length=2,
        restart_prob=0.5,
        seed=7,
    )
    assert np.array_equal(graph1[0], graph2[0])
    assert np.allclose(graph1[1], graph2[1])


def test_weighted_item_adjacency_isolated_nodes():
    user_consumed = {0: [0, 1]}
    item_consumed = {0: [0], 1: [0]}
    indices, values, shape, has_neighbors = build_weighted_item_adjacency(
        user_consumed,
        item_consumed,
        n_items=3,
        neighbor_topk=1,
        num_walks=3,
        walk_length=1,
        restart_prob=0.0,
        seed=11,
    )
    assert shape.tolist() == [3, 3]
    assert len(values) == len(indices)
    assert has_neighbors[2] is False


def test_sampled_bipartite_neighbors_deterministic():
    user_consumed = {0: [0, 1], 1: [1, 2], 2: [2]}
    item_consumed = {0: [0], 1: [0, 1], 2: [1, 2]}
    neighbors = build_bipartite_neighbors(user_consumed, item_consumed, 3, 3)

    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    graph1 = sample_bipartite_neighbors(
        neighbors=neighbors,
        n_users=3,
        user_roots=np.array([0, 1], dtype=np.int32),
        item_roots=np.array([1], dtype=np.int32),
        sample_sizes=(1, 1),
        rng=rng1,
    )
    graph2 = sample_bipartite_neighbors(
        neighbors=neighbors,
        n_users=3,
        user_roots=np.array([0, 1], dtype=np.int32),
        item_roots=np.array([1], dtype=np.int32),
        sample_sizes=(1, 1),
        rng=rng2,
    )

    for part1, part2 in zip(graph1, graph2):
        assert np.array_equal(part1, part2)


def test_sampled_bipartite_neighbors_isolated_root():
    neighbors = build_bipartite_neighbors(
        user_consumed={0: [0]},
        item_consumed={0: [0]},
        n_users=2,
        n_items=2,
    )
    (
        indices,
        values,
        sampled_user_nodes,
        sampled_item_nodes,
        node_has_neighbors,
        user_root_positions,
        item_root_positions,
    ) = sample_bipartite_neighbors(
        neighbors=neighbors,
        n_users=2,
        user_roots=np.array([1], dtype=np.int32),
        item_roots=np.array([1], dtype=np.int32),
        sample_sizes=(2,),
        rng=np.random.default_rng(11),
    )

    assert indices.shape == (0, 2)
    assert values.shape == (0,)
    assert sampled_user_nodes.tolist() == [1]
    assert sampled_item_nodes.tolist() == [1]
    assert node_has_neighbors.tolist() == [False, False]
    assert user_root_positions.tolist() == [0]
    assert item_root_positions.tolist() == [0]
