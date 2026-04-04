Recora
==============

.. image:: https://img.shields.io/github/actions/workflow/status/samousavizade/MyRec/wheels.yml?branch=master
   :target: https://github.com/samousavizade/MyRec/actions/workflows/wheels.yml
   :alt: Build status

.. image:: https://readthedocs.org/projects/recora/badge/?version=stable
    :target: https://recora.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. image:: https://github.com/samousavizade/MyRec/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/samousavizade/MyRec/actions/workflows/ci.yml
   :alt: CI status

.. image:: https://codecov.io/gh/samousavizade/MyRec/branch/master/graph/badge.svg?token=BYOYFBUJRL
   :target: https://codecov.io/gh/samousavizade/MyRec
   :alt: Codecov status

.. image:: https://img.shields.io/pypi/v/recora?color=blue
   :target: https://pypi.org/project/recora/
   :alt: Pypi version

.. image:: https://static.pepy.tech/personalized-badge/recora?period=total&units=international_system&left_color=grey&right_color=lightgrey&left_text=Downloads
   :target: https://pepy.tech/project/recora
   :alt: Downloads

.. image:: https://app.codacy.com/project/badge/Grade/860f0cb5339c41fba9bee5770d09be47
   :target: https://www.codacy.com/gh/samousavizade/MyRec/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=samousavizade/MyRec&amp;utm_campaign=Badge_Grade
   :alt: Codacy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json
   :target: https://github.com/charliermarsh/ruff
   :alt: Ruff

.. image:: https://img.shields.io/github/license/samousavizade/MyRec?color=ff69b4
   :target: https://github.com/samousavizade/MyRec/blob/master/LICENSE
   :alt: License

------------------

**Recora** is an easy-to-use recommender system focused on end-to-end recommendation process.
It contains a training(`recora <https://github.com/samousavizade/MyRec/tree/master/recora>`_) and serving(`libserving <https://github.com/samousavizade/MyRec/tree/master/libserving>`_)
module to let users quickly train and deploy different kinds of recommendation models.

**The main features are:**

+ Implements a number of popular recommendation algorithms such as FM, DIN, BPR etc. See `full algorithm list <https://github.com/samousavizade/MyRec#references>`_.
+ A hybrid recommender system, which allows users to use either collaborative-filtering or content-based features. New features can be added on the fly.
+ Low memory usage, automatically convert categorical and multi-value categorical features to sparse representation.
+ Supports training for both explicit and implicit datasets, as well as negative sampling on implicit data.
+ Provides end-to-end workflow, i.e. data handling / preprocessing -> model training -> evaluate -> save/load -> serving.
+ Supports cold-start prediction and recommendation.
+ Supports dynamic feature and sequence recommendation.
+ Provides unified and friendly API for all algorithms.
+ Easy to retrain model with new users/items from new data.

Quick Start
-----------

The two tabs below demonstrate the process of train, evaluate, predict, recommend and cold-start.

1. **Pure** example(collaborative filtering), which uses ``BPR`` model.

2. **Feat** example(use features), which uses ``YouTubeRanking`` model.

.. tab:: pure_example

    .. literalinclude:: ../../examples/pure_example.py
       :caption: From file `examples/pure_example.py <https://github.com/samousavizade/MyRec/blob/master/examples/pure_example.py>`_
       :name: pure_example.py
       :lines: 15-

.. tab:: feat_example

    .. literalinclude:: ../../examples/feat_example.py
       :caption: From file `examples/feat_example.py <https://github.com/samousavizade/MyRec/blob/master/examples/feat_example.py>`_
       :name: feat_example.py
       :lines: 10-


.. toctree::
   :maxdepth: 1
   :caption: Intro
   :hidden:

   installation
   tutorial

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/data_processing
   user_guide/feature_engineering
   user_guide/model_train
   user_guide/evaluation_save_load
   user_guide/recommendation
   user_guide/embedding
   user_guide/model_retrain

.. toctree::
   :maxdepth: 1
   :caption: Deploy
   :hidden:

   serving_guide/python
   serving_guide/rust
   serving_guide/online

.. toctree::
   :maxdepth: 1
   :caption: Internal
   :hidden:

   internal/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/data/index
   api/algorithms/index
   api/evaluation
   api/serialization

.. toctree::
   :maxdepth: 1
   :caption: Outro
   :hidden:

   Docker <https://github.com/samousavizade/MyRec/tree/master/docker>
   Github <https://github.com/samousavizade/MyRec>

Indices and tables
..................

* :ref:`genindex`
