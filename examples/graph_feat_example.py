import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from recora.algorithms import GraphSage, PinSage
from recora.data import DatasetFeat, split_by_ratio_chrono


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 20, name, "=" * 20)


if __name__ == "__main__":
    data = pd.read_csv("./tests/sample_data/sample_movielens_merged.csv")
    data["item_dense_feat"] = np.random.default_rng(42).normal(size=len(data))

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age", "item_dense_feat"],
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3", "item_dense_feat"],
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    metrics = ["loss", "roc_auc", "precision", "recall", "ndcg"]

    reset_state("GraphSage")
    graphsage = GraphSage(
        "ranking",
        data_info,
        loss_type="softmax",
        embed_size=16,
        layer_sizes=(16, 16),
        n_epochs=2,
        lr=1e-3,
        batch_size=256,
    )
    graphsage.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", graphsage.predict(user=1, item=2333))
    print("recommendation: ", graphsage.recommend_user(user=1, n_rec=7))
    print(
        "dynamic recommendation: ",
        graphsage.recommend_user(
            user=1,
            n_rec=7,
            user_feats={"sex": "M", "age": 25},
            seq=[1, 2, 3],
        ),
    )

    reset_state("PinSage")
    pinsage = PinSage(
        "ranking",
        data_info,
        loss_type="softmax",
        embed_size=16,
        layer_sizes=(16, 16),
        neighbor_topk=10,
        n_epochs=2,
        lr=1e-3,
        batch_size=256,
    )
    pinsage.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", pinsage.predict(user=1, item=2333))
    print("recommendation: ", pinsage.recommend_user(user=1, n_rec=7))
    print(
        "dynamic recommendation: ",
        pinsage.recommend_user(
            user=1,
            n_rec=7,
            user_feats={"sex": "M", "age": 25},
            seq=[1, 2, 3],
        ),
    )
