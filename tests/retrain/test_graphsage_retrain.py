from pathlib import Path

import pandas as pd
import tensorflow as tf

from recora.algorithms import GraphSage
from recora.data import DataInfo, DatasetFeat, split_by_ratio_chrono
from tests.utils_data import SAVE_PATH, remove_path


def test_graphsage_retrain():
    tf.compat.v1.reset_default_graph()
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_merged.csv"
    all_data = pd.read_csv(data_path, sep=",", header=0)
    first_half = all_data[: (len(all_data) // 2)]
    train_part, _ = split_by_ratio_chrono(first_half, test_size=0.2)

    train_data, data_info = DatasetFeat.build_trainset(
        train_part,
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3"],
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age"],
        shuffle=False,
    )
    model = GraphSage(
        "ranking",
        data_info,
        loss_type="softmax",
        embed_size=8,
        layer_sizes=(8, 4),
        n_epochs=1,
        lr=1e-4,
        batch_size=512,
    )
    model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    data_info.save(path=SAVE_PATH, model_name="graphsage_model")
    model.save(SAVE_PATH, "graphsage_model", manual=True, inference_only=False)

    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="graphsage_model")
    second_half = all_data[(len(all_data) // 2) : (len(all_data) * 3 // 4)]
    train_part, _ = split_by_ratio_chrono(second_half, test_size=0.2)
    train_data, new_data_info = DatasetFeat.merge_trainset(
        train_part, new_data_info, merge_behavior=True
    )

    new_model = GraphSage(
        "ranking",
        new_data_info,
        loss_type="softmax",
        embed_size=8,
        layer_sizes=(8, 4),
        n_epochs=1,
        lr=1e-4,
        batch_size=512,
    )
    new_model.rebuild_model(SAVE_PATH, "graphsage_model", full_assign=True)
    new_model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    assert len(new_model.recommend_user(second_half.user.iloc[0], 3)[second_half.user.iloc[0]]) == 3
    remove_path(SAVE_PATH)


def test_graphsage_neighbor_sampling_retrain():
    tf.compat.v1.reset_default_graph()
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_merged.csv"
    all_data = pd.read_csv(data_path, sep=",", header=0)
    first_half = all_data[: (len(all_data) // 2)]
    train_part, _ = split_by_ratio_chrono(first_half, test_size=0.2)

    train_data, data_info = DatasetFeat.build_trainset(
        train_part,
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3"],
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age"],
        shuffle=False,
    )
    model = GraphSage(
        "ranking",
        data_info,
        loss_type="softmax",
        embed_size=8,
        layer_sizes=(8, 4),
        neighbor_sampling=True,
        sample_sizes=(3, 2),
        n_epochs=1,
        lr=1e-4,
        batch_size=512,
    )
    model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)
    data_info.save(path=SAVE_PATH, model_name="graphsage_neighbor_model")
    model.save(SAVE_PATH, "graphsage_neighbor_model", manual=True, inference_only=False)

    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="graphsage_neighbor_model")
    second_half = all_data[(len(all_data) // 2) : (len(all_data) * 3 // 4)]
    train_part, _ = split_by_ratio_chrono(second_half, test_size=0.2)
    train_data, new_data_info = DatasetFeat.merge_trainset(
        train_part, new_data_info, merge_behavior=True
    )

    new_model = GraphSage(
        "ranking",
        new_data_info,
        loss_type="softmax",
        embed_size=8,
        layer_sizes=(8, 4),
        neighbor_sampling=True,
        sample_sizes=(3, 2),
        n_epochs=1,
        lr=1e-4,
        batch_size=512,
    )
    new_model.rebuild_model(SAVE_PATH, "graphsage_neighbor_model", full_assign=True)
    new_model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)
    user = second_half.user.iloc[0]
    assert len(new_model.recommend_user(user, 3)[user]) == 3
    remove_path(SAVE_PATH)
