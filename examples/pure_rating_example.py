import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time

import pandas as pd

from recora.algorithms import ItemCF, UserCF
from recora.data import DatasetPure, split_by_ratio_chrono

if __name__ == "__main__":
    start_time = time.perf_counter()
    data = pd.read_csv(
        "sample_data/sample_movielens_rating.dat",
        sep="::",
        names=["user", "item", "label", "time"],
    )

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    print(data_info)

    metrics = ["rmse", "mae", "r2"]

    user_cf = UserCF(
        task="rating",
        data_info=data_info,
        k_sim=20,
        sim_type="cosine",
        mode="invert",
        num_threads=4,
        min_common=1,
    )
    user_cf.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", user_cf.predict(user=1, item=2333))
    print("recommendation: ", user_cf.recommend_user(user=1, n_rec=7))

    item_cf = ItemCF(
        task="rating",
        data_info=data_info,
        k_sim=20,
        sim_type="pearson",
        mode="invert",
        num_threads=1,
        min_common=1,
    )
    item_cf.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        eval_data=eval_data,
        metrics=metrics,
    )
    print("prediction: ", item_cf.predict(user=1, item=2333))
    print("recommendation: ", item_cf.recommend_user(user=1, n_rec=7))

    print(f"total running time: {(time.perf_counter() - start_time):.2f}")
