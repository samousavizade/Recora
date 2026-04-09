import warnings

import numpy as np
from tqdm import tqdm

from .trainer import BaseTrainer
from ..batch import get_batch_loader, get_tf_feeds
from ..evaluation import print_metrics
from ..layers import normalize_embeds
from ..tfops import choose_tf_loss, lr_decay_config, tf, var_list_by_name
from ..tfops.loss import weighted_mean
from ..utils.constants import EmbeddingModels
from ..utils.misc import colorize, time_block


class TensorFlowTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        batch_size,
        sampler,
        num_neg,
        *args,
        **kwargs,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            batch_size,
            sampler,
            num_neg,
        )
        self.sess = model.sess
        self.use_reg = self._check_reg()
        self._check_params()
        self._ensure_sample_weights()
        self._build_train_ops(*args, **kwargs)

    def run(
        self,
        train_data,
        neg_sampling,
        verbose,
        shuffle,
        eval_data,
        metrics,
        k,
        eval_batch_size,
        eval_user_num,
        num_workers,
    ):
        data_loader = get_batch_loader(
            self.model,
            train_data,
            neg_sampling,
            self.batch_size,
            shuffle,
            num_workers,
            self.model.seed,
        )
        try:
            for epoch in range(1, self.n_epochs + 1):
                if self.lr_decay and verbose > 0:
                    print(
                        f"With lr_decay, epoch {epoch} learning rate: "
                        f"{self.sess.run(self.lr)}"
                    )
                with time_block(f"Epoch {epoch}", verbose):
                    disable = True if verbose <= 0 else False
                    train_total_loss = []
                    for batch_data in tqdm(data_loader, desc="train", disable=disable):
                        fetches = (self.loss, self.training_op)
                        feed_dict = get_tf_feeds(self.model, batch_data, is_training=True)
                        train_loss, _ = self.sess.run(fetches, feed_dict)
                        train_total_loss.append(train_loss)

                if verbose > 1:
                    train_loss_str = "train_loss: " + str(
                        round(float(np.mean(train_total_loss)), 4)
                    )
                    print(f"\t {colorize(train_loss_str, 'green')}")
                    # get embedding for evaluation
                    if EmbeddingModels.contains(self.model.model_name):
                        self.model.set_embeddings()
                    print_metrics(
                        model=self.model,
                        neg_sampling=neg_sampling,
                        eval_data=eval_data,
                        metrics=metrics,
                        eval_batch_size=eval_batch_size,
                        k=k,
                        sample_user_num=eval_user_num,
                        seed=self.model.seed,
                    )
                    print("=" * 30)
        finally:
            if hasattr(data_loader, "close"):
                data_loader.close()

    def _build_train_ops(self, **kwargs):
        self.loss = choose_tf_loss(self.model, self.task, self.loss_type)
        if self.use_reg:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.model.data_info.data_size / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches, **kwargs)
        else:
            self.lr, global_steps = self.lr, None

        # https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64
        # According to the official comment, default value of 1e-8 for `epsilon` is generally not good, so here we choose 1e-5.
        # Users can try tuning this hyperparameter when training is unstable.
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=self.epsilon)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def _ensure_sample_weights(self):
        if not hasattr(self.model, "sample_weights"):
            self.model.sample_weights = tf.placeholder(
                tf.float32, shape=[None], name="sample_weights"
            )

    def _check_reg(self):
        if hasattr(self.model, "reg") and self.model.reg is not None:
            return True
        else:
            return False


class YoutubeRetrievalTrainer(TensorFlowTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        batch_size,
        num_sampled_per_batch,
        sampler,
        num_neg=None,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            batch_size,
            sampler,
            num_neg,
            num_sampled_per_batch,
        )

    def _build_train_ops(self, num_sampled_per_batch, **kwargs):
        self.uses_exact_softmax_loss = False
        num_sampled_per_batch = self._resolve_num_sampled_per_batch(
            num_sampled_per_batch
        )
        # By default, `sampled_softmax_loss` and `nce_loss` in tensorflow
        # uses `log_uniform_candidate_sampler` to sample negative items,
        # which may not be suitable in recommendation scenarios.
        labels = tf.reshape(self.model.item_indices, [-1, 1])
        sampled_values = (
            tf.random.uniform_candidate_sampler(
                true_classes=labels,
                num_true=1,
                num_sampled=num_sampled_per_batch,
                unique=True,
                range_max=self.model.n_items,
            )
            if self.sampler == "uniform" and num_sampled_per_batch > 0
            else None
        )

        user_embeds, item_embeds, item_biases = self._get_loss_inputs()
        if num_sampled_per_batch <= 0:
            self.uses_exact_softmax_loss = True
            self.loss = self._build_exact_softmax_loss(
                user_embeds, item_embeds, item_biases
            )
        elif self.loss_type == "nce":
            self.loss = weighted_mean(
                tf.nn.nce_loss(
                    weights=item_embeds,
                    biases=item_biases,
                    labels=labels,
                    inputs=user_embeds,
                    num_sampled=num_sampled_per_batch,
                    num_classes=self.model.n_items,
                    num_true=1,
                    sampled_values=sampled_values,
                    remove_accidental_hits=True,
                    partition_strategy="div",
                ),
                self.model.sample_weights,
            )
        elif self.loss_type == "sampled_softmax":
            self.loss = weighted_mean(
                tf.nn.sampled_softmax_loss(
                    weights=item_embeds,
                    biases=item_biases,
                    labels=labels,
                    inputs=user_embeds,
                    num_sampled=num_sampled_per_batch,
                    num_classes=self.model.n_items,
                    num_true=1,
                    sampled_values=sampled_values,
                    remove_accidental_hits=True,
                    seed=self.model.seed,
                    partition_strategy="div",
                ),
                self.model.sample_weights,
            )
        else:
            raise ValueError("Loss type must either be `nce` or `sampled_softmax`")

        if self.use_reg:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.model.data_info.data_size / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches, **kwargs)
        else:
            global_steps = None

        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=self.epsilon)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def _resolve_num_sampled_per_batch(self, num_sampled_per_batch):
        requested_num_sampled = (
            num_sampled_per_batch
            if num_sampled_per_batch and num_sampled_per_batch > 0
            else self.batch_size
        )
        max_num_sampled = max(0, self.model.n_items - 1)
        effective_num_sampled = min(requested_num_sampled, max_num_sampled)
        self.num_sampled_per_batch = effective_num_sampled

        if effective_num_sampled != requested_num_sampled:
            if max_num_sampled == 0:
                warnings.warn(
                    "YouTubeRetrieval has fewer than 2 items, so sampled losses "
                    "fall back to exact softmax for training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "YouTubeRetrieval reduced `num_sampled_per_batch` from "
                    f"{requested_num_sampled} to {effective_num_sampled} because "
                    f"the item vocabulary only has {self.model.n_items} items.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return effective_num_sampled

    def _build_exact_softmax_loss(self, user_embeds, item_embeds, item_biases):
        logits = tf.matmul(user_embeds, item_embeds, transpose_b=True)
        logits = tf.nn.bias_add(logits, item_biases)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.model.item_indices,
            logits=logits,
        )
        return weighted_mean(losses, self.model.sample_weights)

    def _get_loss_inputs(self):
        item_biases = self.model.item_biases
        if self.model.norm_embed:
            user_embeds, item_embeds = normalize_embeds(
                self.model.user_embeds, self.model.item_embeds, backend="tf"
            )
            return user_embeds, item_embeds, item_biases
        else:
            return self.model.user_embeds, self.model.item_embeds, item_biases


class WideDeepTrainer(TensorFlowTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        batch_size,
        sampler,
        num_neg,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            batch_size,
            sampler,
            num_neg,
        )

    def _build_train_ops(self, **kwargs):
        self.loss = choose_tf_loss(self.model, self.task, self.loss_type)
        if self.use_reg:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.model.data_info.data_size / self.batch_size)
            self.lr["wide"], wide_global_steps = lr_decay_config(
                self.lr["wide"], n_batches, **kwargs
            )
            self.lr["deep"], deep_global_steps = lr_decay_config(
                self.lr["deep"], n_batches, **kwargs
            )
        else:
            wide_global_steps = deep_global_steps = None

        var_dict = var_list_by_name(names=["wide", "deep"])
        # print(f"{colorize('Wide_variables', 'blue')}: {var_dict['wide']}\n"
        #       f"{colorize('Deep_variables', 'blue')}: {var_dict['deep']}")
        wide_optimizer = tf.train.FtrlOptimizer(
            self.lr["wide"], l1_regularization_strength=1e-3
        )
        wide_optimizer_op = wide_optimizer.minimize(
            loss=total_loss, global_step=wide_global_steps, var_list=var_dict["wide"]
        )

        deep_optimizer = tf.train.AdamOptimizer(self.lr["deep"], epsilon=self.epsilon)
        deep_optimizer_op = deep_optimizer.minimize(
            loss=total_loss, global_step=deep_global_steps, var_list=var_dict["deep"]
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([wide_optimizer_op, deep_optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())
