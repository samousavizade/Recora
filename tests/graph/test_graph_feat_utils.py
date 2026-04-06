import numpy as np

from recora.graph import build_weighted_item_adjacency


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
