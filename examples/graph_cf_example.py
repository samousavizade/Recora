import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import tensorflow as tf

from recora.algorithms import LightGCN, NGCF
from recora.data import DatasetPure, split_by_ratio_chrono


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 20, name, "=" * 20)


if __name__ == "__main__":
    data = pd.read_csv(
        "./sample_data/sample_movielens_rating.dat",
        sep="::",
        names=["user", "item", "label", "time"],
    )

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    metrics = ["loss", "roc_auc", "precision", "recall", "ndcg"]

    reset_state("LightGCN")
    lightgcn = LightGCN(
        "ranking",
        data_info,
        loss_type="bpr",
        embed_size=16,
        n_layers=3,
        n_epochs=2,
        lr=1e-3,
        batch_size=256,
        num_neg=1,
    )
    lightgcn.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", lightgcn.predict(user=1, item=2333))
    print("recommendation: ", lightgcn.recommend_user(user=1, n_rec=7))

    reset_state("NGCF")
    ngcf = NGCF(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        layer_sizes=(16, 16, 16),
        node_dropout_rate=0.1,
        message_dropout_rate=0.1,
        n_epochs=2,
        lr=1e-3,
        batch_size=256,
        num_neg=1,
    )
    ngcf.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", ngcf.predict(user=1, item=2333))
    print("recommendation: ", ngcf.recommend_user(user=1, n_rec=7))
