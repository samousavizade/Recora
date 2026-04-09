import path_setup  # noqa: F401

from pathlib import Path

import pandas as pd
import tensorflow as tf

from recora.algorithms import SVD
from recora.data import DatasetPure, split_by_ratio_chrono


EXAMPLE_DIR = Path(__file__).resolve().parent


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 24, name, "=" * 24)


def run_listwise(loss_type, train_data, eval_data, data_info, sample_user, sample_item):
    model = SVD(
        task="ranking",
        data_info=data_info,
        loss_type=loss_type,
        embed_size=16,
        n_epochs=1,
        lr=1e-3,
        batch_size=4096 * 4,
        num_neg=4,
        listwise_num_pos=4,
        sampler="random",
        listnet_temperature=1.0,
        approx_ndcg_temperature=1.0,
    )
    model.fit(
        train_data=train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "pr_auc", "precision", "recall", "ndcg"],
    )
    print(f"{loss_type} prediction:", model.predict(user=sample_user, item=sample_item))
    print(f"{loss_type} recommendation:", model.recommend_user(user=sample_user, n_rec=7))


if __name__ == "__main__":
    data = pd.read_csv(
        EXAMPLE_DIR / "sample_data" / "sample_movielens_rating.dat",
        sep="::",
        engine="python",
        names=["user", "item", "label", "time"],
    )
    # This dataset already has graded labels (1-5), not only binary labels.
    print("label values in raw data:", sorted(data["label"].unique().tolist()))

    sample_user = data.user.iloc[0]
    sample_item = data.item.iloc[0]
    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)

    reset_state("SVD ListNet with binary labels")
    run_listwise(
        loss_type="cross_entropy",
        train_data=train_data,
        eval_data=eval_data,
        data_info=data_info,
        sample_user=sample_user,
        sample_item=sample_item,
    )

    reset_state("SVD ListNet with graded labels")
    run_listwise(
        loss_type="listnet",
        train_data=train_data,
        eval_data=eval_data,
        data_info=data_info,
        sample_user=sample_user,
        sample_item=sample_item,
    )

    reset_state("SVD ApproxNDCG with graded labels")
    run_listwise(
        loss_type="approx_ndcg",
        train_data=train_data,
        eval_data=eval_data,
        data_info=data_info,
        sample_user=sample_user,
        sample_item=sample_item,
    )
