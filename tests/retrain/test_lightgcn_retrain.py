from pathlib import Path

import pandas as pd
import tensorflow as tf

from recora.algorithms import LightGCN
from recora.data import DataInfo, DatasetPure, split_by_ratio_chrono
from tests.utils_data import SAVE_PATH, remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends


def test_lightgcn_retrain():
    tf.compat.v1.reset_default_graph()
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_rating.dat"
    all_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )

    first_half = all_data[: (len(all_data) // 2)]
    train_data, _ = split_by_ratio_chrono(first_half, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)

    model = LightGCN(
        "ranking",
        data_info,
        loss_type="bpr",
        embed_size=8,
        n_layers=2,
        n_epochs=1,
        lr=1e-4,
        batch_size=256,
    )
    model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    data_info.save(path=SAVE_PATH, model_name="lightgcn_model")
    model.save(
        path=SAVE_PATH, model_name="lightgcn_model", manual=True, inference_only=False
    )

    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="lightgcn_model")
    second_half = all_data[(len(all_data) // 2) :]
    train_data_orig, _ = split_by_ratio_chrono(second_half, test_size=0.2)
    train_data, new_data_info = DatasetPure.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )

    new_model = LightGCN(
        "ranking",
        new_data_info,
        loss_type="cross_entropy",
        embed_size=8,
        n_layers=2,
        n_epochs=1,
        lr=1e-4,
        batch_size=256,
    )
    new_model.rebuild_model(
        path=SAVE_PATH, model_name="lightgcn_model", full_assign=True
    )
    new_model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    assert new_model.get_user_embedding().shape[0] == new_data_info.n_users
    ptest_preds(new_model, "ranking", second_half, with_feats=False)
    ptest_recommends(new_model, new_data_info, second_half, with_feats=False)
    remove_path(SAVE_PATH)
