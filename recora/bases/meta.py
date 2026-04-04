from abc import ABCMeta

from ..tfops import rebuild_tf_model


class ModelMeta(ABCMeta):
    def __new__(mcs, cls_name, bases, cls_dict, **kwargs):
        backend = kwargs["backend"] if "backend" in kwargs else "none"
        if bases[0].__name__ == "TfBase" or backend == "tensorflow":
            cls_dict["rebuild_model"] = rebuild_tf_model
        return super().__new__(mcs, cls_name, bases, cls_dict)
