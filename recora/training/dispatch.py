from .tf_trainer import TensorFlowTrainer, WideDeepTrainer, YoutubeRetrievalTrainer
from ..utils.constants import TfTrainModels


def get_trainer(model):
    train_params = {
        "model": model,
        "task": model.task,
        "loss_type": model.loss_type,
        "n_epochs": model.n_epochs,
        "lr": model.lr,
        "lr_decay": model.lr_decay,
        "epsilon": model.epsilon,
        "batch_size": model.batch_size,
        "sampler": model.sampler,
        "num_neg": model.__dict__.get("num_neg"),
    }

    if TfTrainModels.contains(model.model_name):
        if model.model_name == "YouTubeRetrieval":
            train_params["num_sampled_per_batch"] = model.num_sampled_per_batch
            tf_trainer_cls = YoutubeRetrievalTrainer
        elif model.model_name == "WideDeep":
            tf_trainer_cls = WideDeepTrainer
        else:
            tf_trainer_cls = TensorFlowTrainer
        return tf_trainer_cls(**train_params)
    raise ValueError(f"unsupported model for TensorFlow-only runtime: {model.model_name}")
