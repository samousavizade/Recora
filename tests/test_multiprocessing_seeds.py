import multiprocessing
import random

import numpy as np
import pytest

from recora.batch.batch_data import BatchLoader


class DummyBatchData:
    def __init__(self, data_size):
        self.user_indices = np.arange(data_size)
        self.item_indices = np.arange(data_size)
        self.labels = np.ones(data_size, dtype=np.float32)
        self.sparse_indices = None
        self.dense_values = None

    def __getitem__(self, idx):
        return {
            "user": self.user_indices[idx],
            "item": self.item_indices[idx],
            "label": self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


class SeedCollator:
    def __init__(self, same_seed):
        self.seed = 42
        self.worker_seed = 42
        self.same_seed = same_seed
        self.np_rng = None

    def __call__(self, batch):
        self._set_random_seeds()
        py_random = random.randint(0, 2**32 - 1)
        np_random = self.np_rng.integers(0, 2**32 - 1)
        return py_random, np_random, self.worker_seed

    def _set_random_seeds(self):
        if self.np_rng is None or self.same_seed:
            random.seed(self.worker_seed)
            self.np_rng = np.random.default_rng(self.worker_seed)


@pytest.fixture
def get_loader(request):
    data_size = 20
    same_seed = request.param["same_seed"]
    batch_size = request.param["batch_size"]
    num_workers = request.param["num_workers"]
    batch_data = DummyBatchData(data_size)
    collate_fn = SeedCollator(same_seed=same_seed)
    loader = BatchLoader(
        batch_data=batch_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        seed=42,
    )
    return loader, same_seed, num_workers


@pytest.mark.parametrize(
    "get_loader",
    [
        {"same_seed": True, "num_workers": 4, "batch_size": 3},
        {"same_seed": True, "num_workers": 4, "batch_size": 1},
        {"same_seed": True, "num_workers": 2, "batch_size": 3},
        {"same_seed": True, "num_workers": 2, "batch_size": 1},
        {"same_seed": False, "num_workers": 4, "batch_size": 3},
        {"same_seed": False, "num_workers": 4, "batch_size": 1},
        {"same_seed": False, "num_workers": 2, "batch_size": 3},
        {"same_seed": False, "num_workers": 2, "batch_size": 1},
    ],
    indirect=True,
)
def test_multiprocessing_seeds(get_loader):
    loader, same_seed, num_workers = get_loader
    cpu_cores = multiprocessing.cpu_count()
    if num_workers < cpu_cores:
        py_random, np_random, worker_seeds = [], [], []
        for data in loader:
            py_random.append(data[0])
            np_random.append(data[1])
            worker_seeds.append(data[2])

        if same_seed:
            assert len(np.unique(py_random)) == num_workers
            assert len(np.unique(np_random)) == num_workers
            assert len(np.unique(worker_seeds)) == num_workers
        else:
            assert len(np.unique(py_random)) == len(py_random)
            assert len(np.unique(np_random)) == len(np_random)
            assert len(np.unique(worker_seeds)) == num_workers
