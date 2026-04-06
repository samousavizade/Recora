from types import SimpleNamespace

import numpy as np
import pytest

from recora.algorithms import BPR
from recora.data import DatasetPure, split_by_ratio_chrono
from recora.tfops import tf
from recora.tfops.loss import choose_tf_loss
from recora.training.tf_trainer import YoutubeRetrievalTrainer


def test_choose_tf_loss_all_ones_matches_default():
    tf.reset_default_graph()
    labels = tf.constant([1.0, 0.0], dtype=tf.float32)
    logits = tf.constant([0.0, 0.0], dtype=tf.float32)
    weighted_model = SimpleNamespace(
        model_name="Dummy",
        labels=labels,
        output=logits,
        sample_weights=tf.constant([1.0, 1.0], dtype=tf.float32),
    )
    unweighted_model = SimpleNamespace(model_name="Dummy", labels=labels, output=logits)

    with tf.Session() as sess:
        weighted_loss, unweighted_loss = sess.run(
            [
                choose_tf_loss(weighted_model, "ranking", "cross_entropy"),
                choose_tf_loss(unweighted_model, "ranking", "cross_entropy"),
            ]
        )

    assert weighted_loss == pytest.approx(unweighted_loss)


def test_choose_tf_loss_weighted_cross_entropy_and_zero_weights():
    tf.reset_default_graph()
    labels = tf.constant([1.0, 0.0], dtype=tf.float32)
    logits = tf.constant([0.0, 0.0], dtype=tf.float32)
    weighted_model = SimpleNamespace(
        model_name="Dummy",
        labels=labels,
        output=logits,
        sample_weights=tf.constant([2.0, 0.0], dtype=tf.float32),
    )
    zero_weight_model = SimpleNamespace(
        model_name="Dummy",
        labels=labels,
        output=logits,
        sample_weights=tf.constant([0.0, 0.0], dtype=tf.float32),
    )

    with tf.Session() as sess:
        weighted_loss, zero_loss = sess.run(
            [
                choose_tf_loss(weighted_model, "ranking", "cross_entropy"),
                choose_tf_loss(zero_weight_model, "ranking", "cross_entropy"),
            ]
        )

    assert weighted_loss == pytest.approx(np.log(2.0))
    assert zero_loss == pytest.approx(0.0)


def test_choose_tf_loss_weighted_bpr():
    tf.reset_default_graph()
    model = SimpleNamespace(
        model_name="Dummy",
        bpr_loss=tf.math.log_sigmoid(tf.constant([0.0, 0.0], dtype=tf.float32)),
        sample_weights=tf.constant([3.0, 0.0], dtype=tf.float32),
    )

    with tf.Session() as sess:
        loss = sess.run(choose_tf_loss(model, "ranking", "bpr"))

    assert loss == pytest.approx(np.log(2.0))


def test_choose_tf_loss_weighted_generic_bpr():
    tf.reset_default_graph()
    model = SimpleNamespace(
        model_name="Dummy",
        output=tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
        sample_weights=tf.constant([3.0, 0.0], dtype=tf.float32),
    )

    with tf.Session() as sess:
        loss = sess.run(choose_tf_loss(model, "ranking", "bpr"))

    assert loss == pytest.approx(np.log(2.0))


def test_choose_tf_loss_weighted_ranknet_and_zero_weights():
    tf.reset_default_graph()
    weighted_model = SimpleNamespace(
        model_name="Dummy",
        output=tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
        sample_weights=tf.constant([2.0, 0.0], dtype=tf.float32),
    )
    zero_weight_model = SimpleNamespace(
        model_name="Dummy",
        output=tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
        sample_weights=tf.constant([0.0, 0.0], dtype=tf.float32),
    )

    with tf.Session() as sess:
        weighted_loss, zero_loss = sess.run(
            [
                choose_tf_loss(weighted_model, "ranking", "ranknet"),
                choose_tf_loss(zero_weight_model, "ranking", "ranknet"),
            ]
        )

    assert weighted_loss == pytest.approx(np.log(2.0))
    assert zero_loss == pytest.approx(0.0)


def test_choose_tf_loss_weighted_lambdarank_and_zero_weights():
    tf.reset_default_graph()
    weighted_model = SimpleNamespace(
        model_name="Dummy",
        output=tf.constant([2.0, 2.0, 1.0, 3.0], dtype=tf.float32),
        sample_weights=tf.constant([2.0, 0.0], dtype=tf.float32),
        num_neg=1,
    )
    zero_weight_model = SimpleNamespace(
        model_name="Dummy",
        output=tf.constant([2.0, 2.0, 1.0, 3.0], dtype=tf.float32),
        sample_weights=tf.constant([0.0, 0.0], dtype=tf.float32),
        num_neg=1,
    )
    expected_delta_ndcg = 1.0 - 1.0 / np.log2(3.0)
    expected_loss = np.log1p(np.exp(-1.0)) * expected_delta_ndcg

    with tf.Session() as sess:
        weighted_loss, zero_loss = sess.run(
            [
                choose_tf_loss(weighted_model, "ranking", "lambdarank"),
                choose_tf_loss(zero_weight_model, "ranking", "lambdarank"),
            ]
        )

    assert weighted_loss == pytest.approx(expected_loss)
    assert zero_loss == pytest.approx(0.0)


def test_choose_tf_loss_weighted_softmax():
    tf.reset_default_graph()
    model = SimpleNamespace(
        model_name="Dummy",
        user_embeds=tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32),
        item_embeds=tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32),
        sample_weights=tf.constant([5.0, 0.0], dtype=tf.float32),
        adjust_logits=lambda logits, all_adjust=True: logits,
    )

    with tf.Session() as sess:
        loss = sess.run(choose_tf_loss(model, "ranking", "softmax"))

    assert loss == pytest.approx(np.log1p(np.exp(-1.0)))


@pytest.mark.parametrize("loss_type", ["nce", "sampled_softmax"])
def test_youtube_retrieval_weighted_losses_are_finite(loss_type):
    tf.reset_default_graph()
    model = SimpleNamespace(
        model_name="YouTubeRetrieval",
        sess=tf.Session(),
        data_info=SimpleNamespace(data_size=2),
        n_items=4,
        item_indices=tf.placeholder(tf.int64, shape=[None]),
        user_embeds=tf.Variable([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32),
        item_embeds=tf.Variable(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]], dtype=tf.float32
        ),
        item_biases=tf.Variable(tf.zeros([4], dtype=tf.float32)),
        seed=42,
    )

    trainer = YoutubeRetrievalTrainer(
        model=model,
        task="ranking",
        loss_type=loss_type,
        n_epochs=1,
        lr=1e-3,
        lr_decay=False,
        epsilon=1e-5,
        batch_size=2,
        num_sampled_per_batch=1,
        sampler="uniform",
        num_neg=None,
    )

    non_zero_loss = model.sess.run(
        trainer.loss,
        feed_dict={
            model.item_indices: np.array([0, 1], dtype=np.int64),
            model.sample_weights: np.array([1.0, 1.0], dtype=np.float32),
        },
    )
    zero_loss = model.sess.run(
        trainer.loss,
        feed_dict={
            model.item_indices: np.array([0, 1], dtype=np.int64),
            model.sample_weights: np.array([0.0, 0.0], dtype=np.float32),
        },
    )
    model.sess.close()

    assert np.isfinite(non_zero_loss)
    assert zero_loss == pytest.approx(0.0)


def test_non_tf_training_rejects_non_default_sample_weights(pure_data_small):
    pd_data, _, _, _ = pure_data_small
    train_df, _ = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_df["sample_weight"] = 1.0
    train_df.loc[train_df.index[0], "sample_weight"] = 2.0
    train_data, data_info = DatasetPure.build_trainset(train_df, shuffle=False)

    with pytest.raises(ValueError, match="sample_weight"):
        BPR(task="ranking", data_info=data_info, use_tf=False).fit(
            train_data, neg_sampling=True
        )
