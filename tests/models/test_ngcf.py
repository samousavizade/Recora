import tensorflow as tf
import pytest

from recora.algorithms import NGCF
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


def test_ngcf_invalid_params(prepare_pure_data):
    _, _, _, data_info = prepare_pure_data

    with pytest.raises(ValueError):
        NGCF("rating", data_info)
    with pytest.raises(ValueError):
        NGCF("ranking", data_info, loss_type="softmax")
    with pytest.raises(ValueError):
        NGCF("ranking", data_info, layer_sizes=(16, 0))


@pytest.mark.parametrize(
    "loss_type, neg_sampling",
    [("ranknet", True), ("cross_entropy", True)],
)
def test_ngcf_fit_and_save_load(pure_data_small, loss_type, neg_sampling):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = pure_data_small

    model = NGCF(
        task="ranking",
        data_info=data_info,
        loss_type=loss_type,
        embed_size=8,
        layer_sizes=(8, 4),
        node_dropout_rate=0.1,
        message_dropout_rate=0.1,
        n_epochs=1,
        lr=1e-4,
        batch_size=32,
        sampler="random",
        num_neg=1,
    )
    model.fit(train_data, neg_sampling=neg_sampling, verbose=2, shuffle=True)

    assert model.embedding_dim == 20
    ptest_tf_variables(model)
    ptest_preds(model, "ranking", pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=False)

    loaded_model, loaded_data_info = save_load_model(NGCF, model, data_info)
    ptest_preds(loaded_model, "ranking", pd_data, with_feats=False)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
    with pytest.raises(RuntimeError):
        loaded_model.fit(train_data, neg_sampling=neg_sampling)
