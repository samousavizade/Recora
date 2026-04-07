from .version import tf


def maybe_build_pairwise_scores(model, loss_type):
    if loss_type not in ("bpr", "ranknet", "lambdarank"):
        return
    if hasattr(model, "pairwise_logits") or not hasattr(model, "output"):
        return
    split_index = tf.shape(model.output)[0] // 2
    model.output_pos = model.output[:split_index]
    model.output_neg = model.output[split_index:]
    model.pairwise_logits = model.output_pos - model.output_neg


def choose_tf_loss(model, task, loss_type):
    if task == "rating":
        per_example_loss = tf.math.squared_difference(model.labels, model.output)
        loss = weighted_mean(per_example_loss, get_sample_weights(model, per_example_loss))
    else:
        maybe_build_pairwise_scores(model, loss_type)
        if loss_type == "cross_entropy":
            assert hasattr(model, "output"), (
                f"Binary cross entropy loss is unavailable in `{model.model_name}`"
            )  # fmt: skip
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=model.labels, logits=model.output
            )
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
        elif loss_type == "ranknet":
            assert hasattr(model, "pairwise_logits"), (
                f"RankNet loss is unavailable in `{model.model_name}`"
            )  # fmt: skip
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(model.pairwise_logits),
                logits=model.pairwise_logits,
            )
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
        elif loss_type == "lambdarank":
            assert hasattr(model, "pairwise_logits"), (
                f"LambdaRank loss is unavailable in `{model.model_name}`"
            )  # fmt: skip
            per_example_loss = sampled_lambdarank_loss(
                model.output_pos, model.output_neg, model.pairwise_logits, model.num_neg
            )
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
        elif loss_type == "bpr":
            if hasattr(model, "pairwise_logits"):
                per_example_loss = -tf.math.log_sigmoid(model.pairwise_logits)
            else:
                assert hasattr(model, "bpr_loss"), (
                    f"Bpr loss is unavailable in {model.model_name}"
                )  # fmt: skip
                per_example_loss = -model.bpr_loss
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
        elif loss_type == "focal":
            per_example_loss = focal_loss(labels=model.labels, logits=model.output)
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
        elif loss_type == "max_margin":
            per_example_loss = max_margin_loss(
                model.user_embeds,
                model.item_embeds,
                model.item_embeds_neg,
                model.margin,
            )
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
        elif loss_type == "softmax":
            per_example_loss = softmax_cross_entropy(
                model, model.user_embeds, model.item_embeds
            )
            loss = weighted_mean(
                per_example_loss, get_sample_weights(model, per_example_loss)
            )
            if hasattr(model, "ssl_pattern") and model.ssl_pattern is not None:
                ssl_per_example_loss = softmax_cross_entropy(
                    model,
                    model.ssl_left_embeds,
                    model.ssl_right_embeds,
                    all_adjust=False,
                )
                ssl_loss = weighted_mean(
                    ssl_per_example_loss,
                    get_sample_weights(model, ssl_per_example_loss),
                )
                loss += model.alpha * ssl_loss
        elif loss_type == "listnet":
            per_group_loss = listnet_loss(
                model.output,
                model.labels,
                model.num_neg,
                getattr(model, "listnet_temperature", 1.0),
            )
            loss = weighted_mean(
                per_group_loss,
                get_listwise_sample_weights(model, per_group_loss, model.num_neg),
            )
        elif loss_type == "approx_ndcg":
            per_group_loss = approx_ndcg_loss(
                model.output,
                model.labels,
                model.num_neg,
                getattr(model, "approx_ndcg_temperature", 1.0),
            )
            loss = weighted_mean(
                per_group_loss,
                get_listwise_sample_weights(model, per_group_loss, model.num_neg),
            )
        else:
            raise ValueError(f"unknown loss_type for ranking: {loss_type}")

    return loss


def get_sample_weights(model, losses):
    if hasattr(model, "sample_weights"):
        return model.sample_weights
    return tf.ones_like(losses, dtype=tf.float32)


def weighted_mean(losses, sample_weights, eps=1e-8):
    losses = tf.cast(losses, tf.float32)
    sample_weights = tf.cast(sample_weights, tf.float32)
    numerator = tf.reduce_sum(losses * sample_weights)
    denominator = tf.maximum(
        tf.reduce_sum(sample_weights), tf.constant(eps, dtype=tf.float32)
    )
    return numerator / denominator


def get_listwise_sample_weights(model, losses, num_neg):
    if not hasattr(model, "sample_weights"):
        return tf.ones_like(losses, dtype=tf.float32)
    sample_weights = tf.reshape(model.sample_weights, [-1, num_neg + 1])
    return tf.reduce_mean(sample_weights, axis=1)


# focal loss for binary cross entropy based on [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf)
def focal_loss(labels, logits, alpha=0.25, gamma=2.0):
    weighting_factor = (labels * alpha) + ((1 - labels) * (1 - alpha))
    probs = tf.sigmoid(logits)
    p_t = (labels * probs) + ((1 - labels) * (1 - probs))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return weighting_factor * modulating_factor * bce


def max_margin_loss(user_embeds, item_embeds, item_embeds_neg, margin):
    pos_scores = tf.reduce_sum(user_embeds * item_embeds, axis=1)
    neg_scores = tf.reduce_sum(user_embeds * item_embeds_neg, axis=1)
    return tf.nn.relu(margin + neg_scores - pos_scores)


def softmax_cross_entropy(model, user_embeds, item_embeds, all_adjust=True):
    logits = tf.matmul(user_embeds, item_embeds, transpose_b=True)
    logits = model.adjust_logits(logits, all_adjust)
    labels = tf.range(tf.shape(user_embeds)[0])
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def listnet_loss(scores, labels, num_neg, temperature):
    scores, labels = reshape_sampled_listwise_inputs(scores, labels, num_neg)
    temperature = validate_listwise_temperature(temperature, "listnet_temperature")
    logits = scores / temperature
    target_probs = tf.nn.softmax(labels, axis=1)
    pred_log_probs = tf.nn.log_softmax(logits, axis=1)
    return -tf.reduce_sum(target_probs * pred_log_probs, axis=1)


def approx_ndcg_loss(scores, labels, num_neg, temperature):
    scores, labels = reshape_sampled_listwise_inputs(scores, labels, num_neg)
    temperature = validate_listwise_temperature(
        temperature, "approx_ndcg_temperature"
    )
    gains = tf.pow(2.0, labels) - 1.0
    approx_ranks = approximate_ranks(scores, temperature)
    discounts = _discount(approx_ranks)
    dcg = tf.reduce_sum(gains * discounts, axis=1)

    sorted_gains = tf.sort(gains, axis=1, direction="DESCENDING")
    list_size = tf.shape(scores)[1]
    ideal_ranks = tf.cast(tf.range(list_size) + 1, tf.float32)[tf.newaxis, :]
    idcg = tf.reduce_sum(sorted_gains * _discount(ideal_ranks), axis=1)
    approx_ndcg = tf.math.divide_no_nan(dcg, idcg)
    return tf.where(idcg > 0.0, 1.0 - approx_ndcg, tf.zeros_like(approx_ndcg))


def reshape_sampled_listwise_inputs(scores, labels, num_neg):
    if not num_neg or num_neg < 1:
        raise ValueError("Sampled listwise losses require `num_neg` to be a positive integer")
    list_size = num_neg + 1
    scores = tf.reshape(scores, [-1, list_size])
    labels = tf.reshape(labels, [-1, list_size])
    return scores, labels


def validate_listwise_temperature(temperature, name):
    if temperature <= 0.0:
        raise ValueError(f"`{name}` must be positive, got `{temperature}`")
    return tf.constant(temperature, dtype=tf.float32)


def approximate_ranks(scores, temperature):
    score_i = scores[:, :, tf.newaxis]
    score_j = scores[:, tf.newaxis, :]
    diff = (score_j - score_i) / temperature
    pairwise_probs = tf.sigmoid(diff)
    mask = 1.0 - tf.eye(tf.shape(scores)[1], dtype=tf.float32)[tf.newaxis, :, :]
    return 1.0 + tf.reduce_sum(pairwise_probs * mask, axis=2)


def sampled_lambdarank_loss(output_pos, output_neg, pairwise_logits, num_neg):
    if not num_neg or num_neg < 1:
        raise ValueError("`lambdarank` requires `num_neg` to be a positive integer")

    pos_scores = tf.reshape(output_pos, [-1, num_neg])
    neg_scores = tf.reshape(output_neg, [-1, num_neg])
    pairwise_logits = tf.reshape(pairwise_logits, [-1, num_neg])

    # Each training group contains one positive item and its sampled negatives.
    group_pos_scores = tf.reduce_mean(pos_scores, axis=1, keepdims=True)
    group_scores = tf.concat([group_pos_scores, neg_scores], axis=1)

    sorted_indices = tf.argsort(group_scores, axis=1, direction="DESCENDING")
    ranks = tf.cast(tf.argsort(sorted_indices, axis=1) + 1, tf.float32)
    pos_ranks = ranks[:, :1]
    neg_ranks = ranks[:, 1:]
    delta_ndcg = tf.abs(_discount(pos_ranks) - _discount(neg_ranks))

    pairwise_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(pairwise_logits), logits=pairwise_logits
    )
    return tf.reshape(pairwise_loss * delta_ndcg, [-1])


def _discount(ranks):
    return tf.math.divide_no_nan(
        tf.ones_like(ranks), tf.math.log(ranks + 1.0) / tf.math.log(2.0)
    )
