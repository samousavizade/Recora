"""Utility Functions for Evaluating Data."""
import functools
import math
import numbers

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score

from .computation import (
    build_eval_transformed_data,
    compute_preds,
    compute_probs,
    compute_recommends,
)
from .metrics import (
    LISTWISE_METRICS,
    POINTWISE_METRICS,
    RANKING_METRICS,
    RATING_METRICS,
    average_precision_at_k,
    balanced_accuracy,
    listwise_scores,
    ndcg_at_k,
    pr_auc_score,
    precision_at_k,
    rec_coverage,
    recall_at_k,
    rmse,
    roc_gauc_score,
)
from ..data import TransformedEvalSet, TransformedSet


def _check_metrics(task, metrics, k):
    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]
    if task == "rating":
        for m in metrics:
            if m not in RATING_METRICS:
                raise ValueError(f"Metrics `{m}` is not suitable for rating task...")
    elif task == "ranking":
        for m in metrics:
            if m not in RANKING_METRICS:
                raise ValueError(f"Metrics `{m}` is not suitable for ranking task...")

    if not isinstance(k, numbers.Integral):
        raise TypeError("`k` must be integer")
    return metrics


def sample_users(data, seed, num):
    np_rng = np.random.default_rng(seed)
    unique_users = list(data.positive_consumed)
    if isinstance(num, numbers.Integral) and 0 < num < len(unique_users):
        users = np_rng.choice(unique_users, num, replace=False).tolist()
    else:
        users = unique_users
    return users


def evaluate(
    model,
    data,
    neg_sampling,
    eval_batch_size=8192,
    metrics=None,
    k=10,
    sample_user_num=None,
    seed=42,
):
    """Evaluate the model on specific data and metrics.

    Parameters
    ----------
    model : Base
        Model for evaluation.
    data : :class:`pandas.DataFrame` or :class:`~recora.data.TransformedEvalSet` or
        :class:`~recora.data.TransformedSet`
        Data to evaluate.
    neg_sampling : bool
        Whether to perform negative sampling for evaluating data.
    eval_batch_size : int, default: 8192
        Batch size used in evaluation.
    metrics : list or None, default: None
        List of metrics for evaluating.
    k : int, default: 10
        Parameter of metrics, e.g. recall at k, ndcg at k
    sample_user_num : int or None, default: None
        Number of users used in evaluating. By default, it will use all the users in eval_data.
        Setting it to a positive number will sample users randomly from eval data.
    seed : int, default: 42
        Random seed.

    Returns
    -------
    eval_results : dict of {str : float}
        Evaluation results for the model and data.

    Examples
    --------
    >>> eval_result = evaluate(model, data, neg_sampling=True, metrics=["roc_auc", "precision", "recall"])
    """
    if not isinstance(data, (pd.DataFrame, TransformedEvalSet, TransformedSet)):
        raise ValueError(
            "`data` must be `pandas.DataFrame`, `TransformedEvalSet` or "
            "`TransformedSet`"
        )
    data = build_eval_transformed_data(model, data, neg_sampling, seed)
    if not metrics:
        metrics = ["loss"]
    metrics = _check_metrics(model.task, metrics, k)
    eval_result = dict()
    if model.task == "rating":
        y_pred, y_true = compute_preds(model, data, eval_batch_size)
        for m in metrics:
            if m in ["rmse", "loss"]:
                eval_result[m] = rmse(y_true, y_pred)
            elif m == "mae":
                eval_result[m] = mean_absolute_error(y_true, y_pred)
            elif m == "r2":
                eval_result[m] = r2_score(y_true, y_pred)
    else:
        if POINTWISE_METRICS.intersection(metrics):
            y_prob, y_true = compute_probs(model, data, eval_batch_size)
            for m in metrics:
                if m in ["log_loss", "loss"]:
                    eval_result[m] = log_loss(y_true, y_prob)
                elif m == "balanced_accuracy":
                    eval_result[m] = balanced_accuracy(y_true, y_prob)
                elif m == "roc_auc":
                    eval_result[m] = roc_auc_score(y_true, y_prob)
                elif m == "roc_gauc":
                    eval_result[m] = roc_gauc_score(y_true, y_prob, data.user_indices)
                elif m == "pr_auc":
                    eval_result[m] = pr_auc_score(y_true, y_prob)
        if LISTWISE_METRICS.intersection(metrics):
            users = sample_users(data, seed, sample_user_num)
            num_batch_users = max(1, math.floor(eval_batch_size / model.n_items))
            y_trues = data.positive_consumed
            y_recos = compute_recommends(model, users, k, num_batch_users)
            for m in metrics:
                if m not in LISTWISE_METRICS:
                    continue
                if m == "coverage":
                    eval_result[m] = rec_coverage(y_recos, users, model.n_items)
                    continue
                elif m == "precision":
                    fn = precision_at_k
                elif m == "recall":
                    fn = recall_at_k
                elif m == "map":
                    fn = average_precision_at_k
                elif m == "ndcg":
                    fn = ndcg_at_k
                # noinspection PyUnboundLocalVariable
                eval_result[m] = listwise_scores(fn, y_trues, y_recos, users, k)

    return eval_result


def _unique_metrics(metrics):
    return list(dict.fromkeys(metrics))


def _train_metrics(task, metrics, loss_name):
    metrics = [loss_name] if not metrics else metrics
    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    train_metrics = []
    for metric in metrics:
        metric = loss_name if metric == "loss" else metric
        if task == "ranking" and metric in LISTWISE_METRICS:
            continue
        train_metrics.append(metric)

    if loss_name not in train_metrics:
        train_metrics.insert(0, loss_name)
    return _unique_metrics(train_metrics)


def _print_metric_lines(prefix, metric_values, loss_name, k):
    for metric_name, value in metric_values.items():
        if metric_name == "loss":
            display_metric = loss_name
        elif metric_name in LISTWISE_METRICS:
            display_metric = f"{metric_name}@{k}"
        else:
            display_metric = metric_name
        str_val = (
            f"{round(value, 2)}%" if metric_name == "coverage" else f"{value:.4f}"
        )
        print(f"\t {prefix} {display_metric}: {str_val}")


def print_metrics(
    model,
    neg_sampling,
    train_data=None,
    eval_data=None,
    metrics=None,
    eval_batch_size=8192,
    k=10,
    sample_user_num=2048,
    seed=42,
    verbose=2,
):
    loss_name = "rmse" if model.task == "rating" else "log_loss"
    metrics_fn = functools.partial(
        evaluate,
        model=model,
        neg_sampling=neg_sampling,
        eval_batch_size=eval_batch_size,
        k=k,
        sample_user_num=sample_user_num,
        seed=seed,
    )

    printed = False
    if verbose >= 3 and train_data is not None:
        train_result = metrics_fn(
            data=train_data, metrics=_train_metrics(model.task, metrics, loss_name)
        )
        _print_metric_lines("train", train_result, loss_name, k)
        printed = printed or bool(train_result)

    if verbose >= 2 and eval_data is not None:
        eval_metrics = metrics_fn(data=eval_data, metrics=metrics)
        _print_metric_lines("eval", eval_metrics, loss_name, k)
        printed = printed or bool(eval_metrics)

    return printed
