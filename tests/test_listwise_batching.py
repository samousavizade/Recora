import numpy as np
import pytest

from recora.algorithms import DIN, TwoTower
from recora.batch.batch_data import BatchData, adjust_batch_size, get_collate_fn
from recora.batch.collators import PointwiseCollator


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
def test_sampled_listwise_collator_uses_pointwise_and_preserves_group_order(
    feat_data_small, loss_type
):
    _, train_data, _, data_info = feat_data_small
    model = DIN("ranking", data_info, loss_type=loss_type, sampler="random", num_neg=2)
    original_data = BatchData(train_data, use_features=True)[[0, 1, 2]]

    collator = get_collate_fn(model, neg_sampling=True)
    assert isinstance(collator, PointwiseCollator)

    batch = collator(original_data)
    assert len(batch.users) == len(batch.items) == len(batch.labels) == 9
    assert np.all(batch.labels[::3] == 1.0)
    assert np.all(batch.labels[1::3] == 0.0)
    assert np.all(batch.labels[2::3] == 0.0)
    assert np.array_equal(
        batch.sample_weights, np.repeat(train_data.sample_weights[[0, 1, 2]], 3)
    )


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
def test_adjust_batch_size_distinguishes_sampled_and_inbatch_listwise(
    feat_data_small, loss_type
):
    _, _, _, data_info = feat_data_small
    sampled_listwise_model = DIN(
        "ranking", data_info, loss_type=loss_type, sampler="random", num_neg=2
    )
    inbatch_listwise_model = TwoTower(
        "ranking", data_info, loss_type="softmax", sampler="random", num_neg=2
    )

    assert adjust_batch_size(sampled_listwise_model, 96) == 32
    assert adjust_batch_size(inbatch_listwise_model, 96) == 48
