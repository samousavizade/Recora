import pytest
import tensorflow as tf

from recora.algorithms import (
    AutoInt,
    Caser,
    DeepFM,
    DIN,
    FM,
    GraphSage,
    LightGCN,
    NCF,
    NGCF,
    PinSage,
    RNN4Rec,
    SIM,
    SVD,
    SVDpp,
    Transformer,
    TwoTower,
    WaveNet,
    WideDeep,
    YouTubeRanking,
)
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model

PURE_LISTWISE_MODELS = (
    SVD,
    SVDpp,
    NCF,
    LightGCN,
    NGCF,
    RNN4Rec,
    Caser,
    WaveNet,
)

FEAT_LISTWISE_MODELS = (
    FM,
    DeepFM,
    WideDeep,
    AutoInt,
    DIN,
    Transformer,
    SIM,
    YouTubeRanking,
    TwoTower,
    GraphSage,
    PinSage,
)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
@pytest.mark.parametrize("model_cls", PURE_LISTWISE_MODELS)
def test_pure_models_accept_listwise_constructor_params(
    pure_data_small, model_cls, loss_type
):
    _, _, _, data_info = pure_data_small
    model = model_cls(
        "ranking",
        data_info,
        loss_type=loss_type,
        listnet_temperature=0.7,
        approx_ndcg_temperature=0.3,
    )

    assert model.loss_type == loss_type
    assert model.listnet_temperature == pytest.approx(0.7)
    assert model.approx_ndcg_temperature == pytest.approx(0.3)
    assert model.all_args["listnet_temperature"] == pytest.approx(0.7)
    assert model.all_args["approx_ndcg_temperature"] == pytest.approx(0.3)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
@pytest.mark.parametrize("model_cls", FEAT_LISTWISE_MODELS)
def test_feat_models_accept_listwise_constructor_params(
    feat_data_small, model_cls, loss_type
):
    _, _, _, data_info = feat_data_small
    model = model_cls(
        "ranking",
        data_info,
        loss_type=loss_type,
        listnet_temperature=0.7,
        approx_ndcg_temperature=0.3,
    )

    assert model.loss_type == loss_type
    assert model.listnet_temperature == pytest.approx(0.7)
    assert model.approx_ndcg_temperature == pytest.approx(0.3)
    assert model.all_args["listnet_temperature"] == pytest.approx(0.7)
    assert model.all_args["approx_ndcg_temperature"] == pytest.approx(0.3)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
@pytest.mark.parametrize("model_cls", PURE_LISTWISE_MODELS)
def test_pure_models_listwise_losses_require_neg_sampling(
    pure_data_small, model_cls, loss_type
):
    _, train_data, _, data_info = pure_data_small
    model = model_cls("ranking", data_info, loss_type=loss_type, n_epochs=1)

    with pytest.raises(ValueError, match="must use negative sampling"):
        model.fit(train_data, neg_sampling=False)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
@pytest.mark.parametrize("model_cls", FEAT_LISTWISE_MODELS)
def test_feat_models_listwise_losses_require_neg_sampling(
    feat_data_small, model_cls, loss_type
):
    _, train_data, _, data_info = feat_data_small
    model = model_cls("ranking", data_info, loss_type=loss_type, n_epochs=1)

    with pytest.raises(ValueError, match="must use negative sampling"):
        model.fit(train_data, neg_sampling=False)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
def test_svd_listwise_smoke(pure_data_small, loss_type):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = pure_data_small
    model = SVD(
        "ranking",
        data_info,
        loss_type=loss_type,
        embed_size=8,
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
        num_neg=2,
    )
    model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)

    ptest_preds(model, "ranking", pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=False)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
def test_deepfm_listwise_smoke(feat_data_small, loss_type):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = feat_data_small
    model = DeepFM(
        "ranking",
        data_info,
        loss_type=loss_type,
        embed_size=8,
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
        num_neg=2,
    )
    model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)

    ptest_preds(model, "ranking", pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=False)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
def test_rnn4rec_listwise_smoke(pure_data_small, loss_type):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = pure_data_small
    model = RNN4Rec(
        "ranking",
        data_info,
        loss_type=loss_type,
        embed_size=4,
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
        num_neg=2,
    )
    model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)

    ptest_tf_variables(model)
    ptest_preds(model, "ranking", pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=False)
    ptest_dyn_recommends(model, pd_data)


@pytest.mark.parametrize("loss_type", ["listnet", "approx_ndcg"])
def test_graphsage_listwise_smoke(feat_data_small, loss_type):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = feat_data_small
    model = GraphSage(
        "ranking",
        data_info,
        loss_type=loss_type,
        embed_size=8,
        layer_sizes=(8, 4),
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
        num_neg=2,
    )
    model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)

    ptest_tf_variables(model)
    ptest_preds(model, "ranking", pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
    ptest_dyn_recommends(model, pd_data)


def test_graphsage_listwise_temperatures_persist_through_save_load(feat_data_small):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, _, data_info = feat_data_small
    model = GraphSage(
        "ranking",
        data_info,
        loss_type="listnet",
        embed_size=8,
        layer_sizes=(8, 4),
        n_epochs=1,
        lr=1e-4,
        batch_size=64,
        num_neg=2,
        listnet_temperature=0.7,
        approx_ndcg_temperature=0.3,
    )
    model.fit(train_data, neg_sampling=True, verbose=0, shuffle=True)

    loaded_model, loaded_data_info = save_load_model(GraphSage, model, data_info)
    assert loaded_model.listnet_temperature == pytest.approx(0.7)
    assert loaded_model.approx_ndcg_temperature == pytest.approx(0.3)
    ptest_preds(loaded_model, "ranking", pd_data, with_feats=False)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
