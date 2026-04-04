import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import tensorflow as tf

from recora.algorithms import BPR  # pure data, algorithm BPR
from recora.data import DatasetPure, random_split
from recora.evaluation import evaluate


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


if __name__ == "__main__":
    reset_state("BPR")
    data = pd.read_csv(
        "sample_data/sample_movielens_rating.dat",
        sep="::",
        names=["user", "item", "label", "time"],
    )

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)
    print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

    bpr = BPR(
        task="ranking",
        data_info=data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=3,
        lr=1e-3,
        batch_size=2048,
        num_neg=1,
    )
    # monitor metrics on eval_data during training
    bpr.fit(
        train_data,
        neg_sampling=True,  # sample negative items for train and eval data
        verbose=2,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )

    # do final evaluation on test data
    print(
        "evaluate_result: ",
        evaluate(
            model=bpr,
            data=test_data,
            neg_sampling=True,  # sample negative items for test data
            metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
        ),
    )
    # predict preference of user 2211 to item 110
    print("prediction: ", bpr.predict(user=2211, item=110))
    # recommend 7 items for user 2211
    print("recommendation: ", bpr.recommend_user(user=2211, n_rec=7))

    # cold-start prediction
    print(
        "cold prediction: ",
        bpr.predict(user="ccc", item="not item", cold_start="average"),
    )
    # cold-start recommendation
    print(
        "cold recommendation: ",
        bpr.recommend_user(user="are we good?", n_rec=7, cold_start="popular"),
    )
