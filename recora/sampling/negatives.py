import math
import random

import numpy as np


def _check_invalid_negatives(negatives, items_pos, items=None):
    if items is not None and len(items) > 0:
        invalid_indices = np.union1d(
            np.where(negatives == items_pos)[0], np.where(negatives == items)[0]
        )
    else:
        invalid_indices = np.where(negatives == items_pos)[0]
    return list(invalid_indices)


def negatives_from_random(
    np_rng, n_items, items_pos, num_neg, items=None, tolerance=10
):
    items_pos = np.repeat(items_pos, num_neg) if num_neg > 1 else items_pos
    items = np.repeat(items, num_neg) if num_neg > 1 and items is not None else items
    replace = False if len(items_pos) < n_items else True
    negatives = np_rng.choice(n_items, size=len(items_pos), replace=replace)
    for _ in range(tolerance):
        invalid_indices = _check_invalid_negatives(negatives, items_pos, items)
        if not invalid_indices:
            break
        negatives[invalid_indices] = np_rng.choice(
            n_items, size=len(invalid_indices), replace=True
        )
    return negatives


def negatives_from_popular(np_rng, n_items, items_pos, num_neg, items=None, probs=None):
    items_pos = np.repeat(items_pos, num_neg) if num_neg > 1 else items_pos
    items = np.repeat(items, num_neg) if num_neg > 1 and items is not None else items
    negatives = np_rng.choice(n_items, size=len(items_pos), replace=True, p=probs)
    invalid_indices = _check_invalid_negatives(negatives, items_pos, items)
    if invalid_indices:
        negatives[invalid_indices] = np_rng.choice(
            n_items, size=len(invalid_indices), replace=True, p=probs
        )
    return negatives


def negatives_from_popular_unconsumed(
    np_rng,
    user_consumed_set,
    users,
    items_pos,
    n_items,
    num_neg,
    probs=None,
    tolerance=10,
    user_consumed_cache=None,
    user_unconsumed_cache=None,
    cache_item_limit=4096,
):
    users = np.asarray(users, dtype=np.int32)
    items_pos = np.asarray(items_pos, dtype=np.int32)
    if num_neg > 1:
        users = np.repeat(users, num_neg)
        items_pos = np.repeat(items_pos, num_neg)

    negatives = np_rng.choice(n_items, size=len(items_pos), replace=True, p=probs)
    all_indices = np.arange(len(items_pos), dtype=np.int32)

    def find_invalid(indices):
        if len(indices) == 0:
            return np.empty(0, dtype=np.int32)

        same_pos = negatives[indices] == items_pos[indices]
        invalid_indices = list(indices[same_pos])
        check_indices = indices[~same_pos]
        if len(check_indices) == 0:
            return np.asarray(invalid_indices, dtype=np.int32)

        check_users = users[check_indices]
        check_items = negatives[check_indices]
        for idx, user_id, sampled_item in zip(check_indices, check_users, check_items):
            if sampled_item in user_consumed_set[user_id]:
                invalid_indices.append(idx)
        return np.asarray(invalid_indices, dtype=np.int32)

    invalid_indices = find_invalid(all_indices)
    for _ in range(tolerance):
        if len(invalid_indices) == 0:
            return negatives
        negatives[invalid_indices] = np_rng.choice(
            n_items, size=len(invalid_indices), replace=True, p=probs
        )
        invalid_indices = find_invalid(invalid_indices)

    # For unresolved examples (usually dense users), switch to exact sampling from
    # user-specific unconsumed items using popularity probabilities.
    unresolved = invalid_indices
    if len(unresolved) == 0:
        return negatives

    unresolved_users = users[unresolved]
    unresolved_groups = {}
    for idx, u in zip(unresolved, unresolved_users):
        unresolved_groups.setdefault(int(u), []).append(int(idx))

    all_items = np.arange(n_items, dtype=np.int32)
    for user_id, idx_list in unresolved_groups.items():
        row_indices = np.asarray(idx_list, dtype=np.int32)
        cached = (
            None
            if user_unconsumed_cache is None
            else user_unconsumed_cache.get(user_id)
        )
        if cached is None:
            consumed = (
                user_consumed_cache[user_id]
                if user_consumed_cache is not None and user_id in user_consumed_cache
                else np.fromiter(
                    user_consumed_set[user_id],
                    dtype=np.int32,
                    count=len(user_consumed_set[user_id]),
                )
            )
            if user_consumed_cache is not None and user_id not in user_consumed_cache:
                user_consumed_cache[user_id] = consumed
            if len(consumed) == 0:
                candidates = all_items
            elif len(consumed) >= n_items:
                candidates = np.empty(0, dtype=np.int32)
            else:
                consumed_mask = np.ones(n_items, dtype=bool)
                consumed_mask[consumed] = False
                candidates = np.flatnonzero(consumed_mask).astype(np.int32, copy=False)
            if len(candidates) > 0:
                cand_probs = probs[candidates] if probs is not None else None
                if cand_probs is not None:
                    cand_prob_sum = float(np.sum(cand_probs))
                    if cand_prob_sum > 0.0:
                        cand_probs = cand_probs / cand_prob_sum
                    else:
                        cand_probs = None
            else:
                cand_probs = None
            if user_unconsumed_cache is not None and 0 < len(candidates) <= cache_item_limit:
                user_unconsumed_cache[user_id] = (candidates, cand_probs)
        else:
            candidates, cand_probs = cached

        # Remove positive items in this unresolved user-group.
        pos_items = np.unique(items_pos[row_indices])
        if len(candidates) > 0 and len(pos_items) > 0:
            keep = ~np.isin(candidates, pos_items, assume_unique=False)
            candidates = candidates[keep]
            if cand_probs is not None:
                cand_probs = cand_probs[keep]
                cand_prob_sum = float(np.sum(cand_probs))
                cand_probs = cand_probs / cand_prob_sum if cand_prob_sum > 0.0 else None

        if len(candidates) == 0:
            sampled = np_rng.choice(n_items, size=len(row_indices), replace=True, p=probs)
            same_pos = sampled == items_pos[row_indices]
            for _ in range(tolerance):
                if not np.any(same_pos):
                    break
                sampled[same_pos] = np_rng.choice(
                    n_items, size=int(np.sum(same_pos)), replace=True, p=probs
                )
                same_pos = sampled == items_pos[row_indices]
            if np.any(same_pos):
                sampled[same_pos] = (items_pos[row_indices][same_pos] + 1) % n_items
            negatives[row_indices] = sampled
        else:
            negatives[row_indices] = np_rng.choice(
                candidates, size=len(row_indices), replace=True, p=cand_probs
            )
    return negatives


def negatives_from_out_batch(np_rng, n_items, items_pos, items, num_neg):
    sample_num = len(items_pos) * num_neg
    candidate_items = list(set(range(n_items)) - set(items_pos) - set(items))
    if not candidate_items:
        return np_rng.choice(n_items, size=sample_num, replace=True)
    replace = False if sample_num < len(candidate_items) else True
    return np_rng.choice(candidate_items, size=sample_num, replace=replace)


def negatives_from_unconsumed(
    user_consumed_set, users, items, n_items, num_neg, tolerance=10
):
    _floor = math.floor
    _random = random.random

    def sample_one():
        return _floor(n_items * _random())

    negatives = []
    for u, i in zip(users, items):
        u_negs = []
        for _ in range(num_neg):
            success = False
            n = sample_one()
            for _ in range(tolerance):
                if n != i and n not in u_negs and n not in user_consumed_set[u]:
                    success = True
                    break
                n = sample_one()
            if not success:
                for _ in range(tolerance):
                    if n != i and n not in u_negs:
                        break
                    n = sample_one()
            u_negs.append(n)
        negatives.extend(u_negs)
    return np.array(negatives)


def neg_probs_from_frequency(item_consumed, n_items, temperature):
    freqs = []
    for i in range(n_items):
        freq = len(set(item_consumed[i]))
        if temperature != 1.0:
            freq = pow(freq, temperature)
        freqs.append(freq)
    freqs = np.array(freqs)
    return freqs / np.sum(freqs)


def pos_probs_from_frequency(item_consumed, n_users, n_items, alpha):
    probs = []
    for i in range(n_items):
        prob = len(set(item_consumed[i])) / n_users
        prob = (math.sqrt(prob / alpha) + 1) * (alpha / prob)
        probs.append(prob)
    return probs
