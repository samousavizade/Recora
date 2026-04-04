import math
import multiprocessing as mp

import numpy as np

from .collators import BaseCollator as NormalCollator
from .collators import PairwiseCollator, PointwiseCollator, SparseCollator, set_worker_seed
from ..utils.constants import FeatModels
from ..utils.misc import transform_seed
from ..utils.validate import is_listwise_training


class BatchData:
    def __init__(self, data, use_features):
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.sparse_indices = data.sparse_indices
        self.dense_values = data.dense_values
        self.use_features = use_features

    def __getitem__(self, idx):
        batch = {
            "user": self.user_indices[idx],
            "item": self.item_indices[idx],
            "label": self.labels[idx],
        }
        if self.use_features and self.sparse_indices is not None:
            batch["sparse"] = self.sparse_indices[idx]
        if self.use_features and self.dense_values is not None:
            batch["dense"] = self.dense_values[idx]
        return batch

    def __len__(self):
        return len(self.labels)


def _worker_main(batch_data, collate_fn, input_queue, output_queue, worker_seed):
    set_worker_seed(collate_fn, worker_seed)
    while True:
        task = input_queue.get()
        if task is None:
            break
        order, batch_indices = task
        try:
            batch = batch_data[batch_indices]
            output_queue.put((order, collate_fn(batch)))
        except BaseException as e:  # pragma: no cover
            output_queue.put((order, e))
            break


class BatchLoader:
    def __init__(self, batch_data, batch_size, shuffle, collate_fn, num_workers, seed):
        self.batch_data = batch_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.seed = seed

    def __len__(self):
        return math.ceil(len(self.batch_data) / self.batch_size)

    def __iter__(self):
        batch_indices = list(self._iter_batch_indices())
        if self.num_workers <= 0:
            for indices in batch_indices:
                yield self.collate_fn(self.batch_data[indices])
            return

        ctx = mp.get_context("spawn")
        output_queue = ctx.Queue()
        input_queues = [ctx.Queue() for _ in range(self.num_workers)]
        workers = [
            ctx.Process(
                target=_worker_main,
                args=(
                    self.batch_data,
                    self.collate_fn,
                    input_queues[worker_id],
                    output_queue,
                    transform_seed(self.seed, worker_id + 1),
                ),
            )
            for worker_id in range(self.num_workers)
        ]

        try:
            for worker in workers:
                worker.start()
            for order, indices in enumerate(batch_indices):
                input_queues[order % self.num_workers].put((order, indices))
            for input_queue in input_queues:
                input_queue.put(None)

            next_order = 0
            buffered = {}
            for _ in range(len(batch_indices)):
                order, result = output_queue.get()
                if isinstance(result, BaseException):  # pragma: no cover
                    raise result
                buffered[order] = result
                while next_order in buffered:
                    yield buffered.pop(next_order)
                    next_order += 1
        finally:
            for worker in workers:
                worker.join(timeout=1)
                if worker.is_alive():  # pragma: no cover
                    worker.terminate()
                    worker.join()

    def _iter_batch_indices(self):
        indices = np.arange(len(self.batch_data))
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]


def get_batch_loader(model, data, neg_sampling, batch_size, shuffle, num_workers, seed):
    use_features = True if FeatModels.contains(model.model_name) else False
    batch_data = BatchData(data, use_features)
    collate_fn = get_collate_fn(model, neg_sampling)
    return BatchLoader(batch_data, batch_size, shuffle, collate_fn, num_workers, seed)


def get_collate_fn(model, neg_sampling):
    model_name, data_info = model.model_name, model.data_info
    separate_features = True if model_name == "TwoTower" else False
    if model_name == "YouTubeRetrieval":
        return SparseCollator(model, data_info)
    if model_name == "TwoTower" and model.loss_type == "softmax":
        return NormalCollator(model, data_info, separate_features)
    if model.task == "rating" or not neg_sampling:
        return NormalCollator(model, data_info, separate_features)
    if model.loss_type in ("cross_entropy", "focal"):
        return PointwiseCollator(model, data_info, separate_features)
    return PairwiseCollator(model, data_info, repeat_positives=True)


def adjust_batch_size(model, original_batch_size):
    if is_listwise_training(model):
        return original_batch_size
    if model.sampler is not None:
        if model.loss_type in ("cross_entropy", "focal"):
            return max(1, int(original_batch_size / (model.num_neg + 1)))
        return max(1, int(original_batch_size / model.num_neg))
    return original_batch_size
