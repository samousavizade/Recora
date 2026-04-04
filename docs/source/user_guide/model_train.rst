Model & Train
=============

Pure and Feat Models
--------------------

Recora is a hybrid recommender system, which means you can choose whether to use
features other than user behaviors or not. For models only use user behaviors, we classify
them as ``pure`` models. This category includes ``UserCF``, ``ItemCF``, ``SVD``, ``SVD++``,
``ALS``, ``NCF``, ``BPR``, ``RNN4Rec``, ``Caser``, ``WaveNet``.

Then for models that can use other features (e.g., age, sex, name etc.), we call
them ``feat`` models. This category includes ``WideDeep``, ``FM``, ``DeepFM``, ``YouTubeRetrieval``,
``YouTubeRanking``, ``AutoInt``, ``DIN``, ``TwoTower``, ``Transformer``, ``SIM``.

The main difference on usage between these two kinds of models are:

1.  ``pure`` models should use :class:`~recora.data.dataset.DatasetPure` to process data,
and ``feat`` models should use :class:`~recora.data.dataset.DatasetFeat`.

2. When using ``feat`` models, four parameters should be provided,
i.e. [``sparse_col``, ``dense_col``, ``user_col``, ``item_col``], as otherwise the model will
have no idea how to deal with all kinds of features.

The ``fit()`` method is the sole method for training a model in Recora.
You can find some typical usages in these examples:

.. SeeAlso::

    + `pure_rating_example.py <https://github.com/samousavizade/MyRec/blob/master/examples/pure_rating_example.py>`_
    + `pure_ranking_example.py <https://github.com/samousavizade/MyRec/blob/master/examples/pure_ranking_example.py>`_
    + `feat_rating_example.py <https://github.com/samousavizade/MyRec/blob/master/examples/feat_rating_example.py>`_
    + `feat_ranking_example.py <https://github.com/samousavizade/MyRec/blob/master/examples/feat_ranking_example.py>`_

In addition, some models can leverage user behavior sequence. These sequence-oriented models
overlap with both ``pure`` and ``feat`` categories, while keeping the same training APIs.

Multiprocess data loading
-------------------------

Most TensorFlow models can enable multiprocess batch loading during training by setting
the ``num_workers`` parameter in ``fit()``. This uses Recora's internal batch loader
to prepare NumPy batches in worker processes before feeding them into TensorFlow.

.. code-block:: python3

    >>> model.fit(train_data, neg_sampling=False, num_workers=2)

Loss
----

Recora provides some options on loss type for *ranking* :ref:`task <Task>`.
The default loss type for most models is *cross entropy* loss. Since version ``0.10.0``,
focal loss was added into the library. First introduced in `Lin et al., 2018 <https://arxiv.org/pdf/1708.02002.pdf>`_,
focal loss down-weights well-classified examples and focuses on hard examples to get better
training performance, and here is the `implementation <https://github.com/samousavizade/MyRec/blob/master/recora/tfops/loss.py#L34>`_.

In order to choose which loss to use, simply set the ``loss_type`` parameter:

.. code-block:: python3

   >>> model = Caser(task="ranking", loss_type="cross_entropy", ...)
   >>> model = Caser(task="ranking", loss_type="focal", ...)

The table below lists the losses and :ref:`negative samplers <negative-samplers>` that can be used for `ranking` task in each algorithm:

+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                      Algorithm                                       |                 Loss                  |                Sampler                 |
+======================================================================================+=======================================+========================================+
|                             UserCF, ItemCF, ALS                                     |                   /                   |                   /                    |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                         BPR                                          |                  bpr                  |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                   YouTubeRetrieval                                   |          sampled_softmax, nce         |             uniform, other             |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
| SVD, SVD++, NCF, Wide&Deep, FM, DeepFM, YouTubeRanking, AutoInt, DIN, Caser, WaveNet |         cross_entropy, focal          |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                       RNN4Rec                                        |       cross_entropy, focal, bpr       |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                       TwoTower                                       |   cross_entropy, max_margin, softmax  |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+

.. caution::

    *bpr* and *max_margin* belong to pairwise loss, so they must be used with negative sampling,
    which means your data should only contains positive samples when using these losses.
