import tensorflow as tf
import pytest
from numpy.testing import assert_array_equal

from recora.algorithms import PinSage
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model


def test_pinsage_invalid_params(feat_data_small):
    _, _, _, data_info = feat_data_small

    with pytest.raises(ValueError):
        PinSage("rating", data_info)
    with pytest.raises(ValueError):
        PinSage("ranking", data_info, loss_type="bpr")
    with pytest.raises(ValueError):
        PinSage("ranking", data_info, neighbor_topk=0)


@pytest.mark.parametrize(
    "loss_type, neg_sampling",
    [("softmax", True), ("cross_entropy", False)],
)
def test_pinsage_fit_and_save_load(feat_data_small, loss_type, neg_sampling):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = feat_data_small
    if not neg_sampling:
        set_ranking_labels(train_data)

    model = PinSage(
        task="ranking",
        data_info=data_info,
        loss_type=loss_type,
        embed_size=8,
        layer_sizes=(8, 4),
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
    )
    model.fit(train_data, neg_sampling=neg_sampling, verbose=2, shuffle=True)

    assert model.embedding_dim == 4
    ptest_tf_variables(model)
    ptest_preds(model, "ranking", pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
    dyn_rec = ptest_dyn_recommends(model, pd_data)
    assert model.dyn_user_embedding(pd_data.user.iloc[0], seq=[]).shape[0] == 4

    loaded_model, loaded_data_info = save_load_model(PinSage, model, data_info)
    ptest_preds(loaded_model, "ranking", pd_data, with_feats=False)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
    loaded_dyn_rec = ptest_dyn_recommends(loaded_model, pd_data)
    assert_array_equal(dyn_rec, loaded_dyn_rec)


def test_pinsage_max_margin(feat_data_small):
    tf.compat.v1.reset_default_graph()
    _, train_data, _, data_info = feat_data_small
    model = PinSage(
        task="ranking",
        data_info=data_info,
        loss_type="max_margin",
        embed_size=8,
        layer_sizes=(8, 4),
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
    )
    model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    assert model.get_item_embedding().shape[1] == model.embedding_dim
