from .version import tf


def choose_tf_loss(model, task, loss_type):
    if task == "rating":
        per_example_loss = tf.math.squared_difference(model.labels, model.output)
        loss = weighted_mean(per_example_loss, get_sample_weights(model, per_example_loss))
    else:
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
        elif loss_type == "bpr":
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
