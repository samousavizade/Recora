import path_setup  # noqa: F401

from pathlib import Path

import pandas as pd
import tensorflow as tf

from recora.algorithms import GraphSage
from recora.data import DatasetFeat, split_by_ratio_chrono


EXAMPLE_DIR = Path(__file__).resolve().parent


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 20, name, "=" * 20)


if __name__ == "__main__":
    data = pd.read_csv(EXAMPLE_DIR / "sample_data" / "sample_movielens_merged.csv")

    train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]

    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        user_col=user_col,
        item_col=item_col,
        sparse_col=sparse_col,
        dense_col=dense_col,
    )
    eval_data = DatasetFeat.build_testset(eval_data)

    sample_user = data.user.iloc[0]
    sample_item = data.item.iloc[0]
    sample_user_feats = {"sex": data.sex.iloc[0], "age": int(data.age.iloc[0])}
    sample_seq = data.item.iloc[:5].tolist()

    reset_state("GraphSage Neighbor Sampling")
    graphsage = GraphSage(
        task="ranking",
        data_info=data_info,
        loss_type="softmax",
        embed_size=16,
        hidden_units=(128, 64),
        layer_sizes=(16, 16),
        neighbor_sampling=True,
        sample_sizes=(10, 5),
        n_epochs=1,
        lr=1e-3,
        batch_size=256,
        num_neg=1,
        sampler="random",
    )
    # This enables paper-style fixed-fanout neighborhood sampling during training.
    # Inference still uses full-graph embeddings for prediction and recommendation.
    graphsage.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "pr_auc", "precision", "recall", "ndcg"],
    )

    print("prediction: ", graphsage.predict(user=sample_user, item=sample_item))
    print("recommendation: ", graphsage.recommend_user(user=sample_user, n_rec=7))
    print("user embedding shape: ", graphsage.get_user_embedding(sample_user).shape)
    print("item embedding shape: ", graphsage.get_item_embedding(sample_item).shape)
    print(
        "dynamic recommendation: ",
        graphsage.recommend_user(
            user=sample_user,
            n_rec=7,
            user_feats=sample_user_feats,
            seq=sample_seq,
        ),
    )
