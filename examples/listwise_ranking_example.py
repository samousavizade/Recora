import path_setup  # noqa: F401

import time
from pathlib import Path

import pandas as pd
import tensorflow as tf

from recora.algorithms import GraphSage, SVD
from recora.data import DatasetFeat, DatasetPure, split_by_ratio_chrono


EXAMPLE_DIR = Path(__file__).resolve().parent


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


if __name__ == "__main__":
    start_time = time.perf_counter()

    pure_data = pd.read_csv(
        EXAMPLE_DIR / "sample_data" / "sample_movielens_rating.dat",
        sep="::",
        engine="python",
        names=["user", "item", "label", "time"],
    )
    pure_user = pure_data.user.iloc[0]
    pure_item = pure_data.item.iloc[0]
    pure_train, pure_eval = split_by_ratio_chrono(pure_data, test_size=0.2)
    pure_train, pure_info = DatasetPure.build_trainset(pure_train)
    pure_eval = DatasetPure.build_evalset(pure_eval)

    metrics = ["loss", "roc_auc", "precision", "recall", "ndcg"]

    reset_state("SVD ListNet")
    svd = SVD(
        "ranking",
        pure_info,
        loss_type="listnet",
        embed_size=16,
        n_epochs=1,
        lr=1e-3,
        batch_size=256,
        num_neg=2,
        sampler="random",
        listnet_temperature=1.0,
    )
    # Sampled listwise losses reuse the standard positive-plus-negatives path.
    svd.fit(
        pure_train,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=pure_eval,
        metrics=metrics,
    )
    print("prediction: ", svd.predict(user=pure_user, item=pure_item))
    print("recommendation: ", svd.recommend_user(user=pure_user, n_rec=7))

    feat_data = pd.read_csv(
        EXAMPLE_DIR / "sample_data" / "sample_movielens_merged.csv",
        sep=",",
        header=0,
    )
    feat_user = feat_data.user.iloc[0]
    feat_item = feat_data.item.iloc[0]
    feat_train, feat_eval = split_by_ratio_chrono(feat_data, test_size=0.2)
    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]
    feat_train, feat_info = DatasetFeat.build_trainset(
        feat_train, user_col, item_col, sparse_col, dense_col
    )
    feat_eval = DatasetFeat.build_testset(feat_eval)

    reset_state("GraphSage ApproxNDCG")
    graphsage = GraphSage(
        "ranking",
        feat_info,
        loss_type="approx_ndcg",
        embed_size=16,
        hidden_units=(128, 64),
        layer_sizes=(16, 16),
        n_epochs=1,
        lr=1e-3,
        batch_size=256,
        num_neg=2,
        sampler="random",
        approx_ndcg_temperature=1.0,
    )
    graphsage.fit(
        feat_train,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=feat_eval,
        metrics=metrics,
    )
    print("prediction: ", graphsage.predict(user=feat_user, item=feat_item))
    print("recommendation: ", graphsage.recommend_user(user=feat_user, n_rec=7))

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
