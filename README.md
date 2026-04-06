# Recora

[Build](https://github.com/samousavizade/MyRec/actions/workflows/wheels.yml)
[CI](https://github.com/samousavizade/MyRec/actions/workflows/ci.yml)
[Codecov](https://app.codecov.io/gh/samousavizade/MyRec)
[pypi](https://pypi.org/project/recora/)
[Downloads](https://pepy.tech/project/recora)
[Codacy Badge](https://www.codacy.com/gh/samousavizade/MyRec/dashboard?utm_source=github.com&utm_medium=referral&utm_content=samousavizade/MyRec&utm_campaign=Badge_Grade)
[Code style: black](https://github.com/psf/black)
[Ruff](https://github.com/charliermarsh/ruff)
[Documentation Status](https://recora.readthedocs.io/en/latest/?badge=latest)
[python versions](https://pypi.org/project/recora/)
[License](https://github.com/samousavizade/MyRec/blob/master/LICENSE)

## Overview

**Recora** is an easy-to-use recommender system focused on end-to-end recommendation process. It contains a training([recora](https://github.com/samousavizade/MyRec/tree/master/recora)) and serving([libserving](https://github.com/samousavizade/MyRec/tree/master/libserving)) module to let users quickly train and deploy different kinds of recommendation models.

**The main features are:**

- Implements a number of popular recommendation algorithms such as FM, DIN, BPR etc. See [full algorithm list](#references).
- A hybrid recommender system, which allows user to use either collaborative-filtering or content-based features. New features can be added on the fly.
- Low memory usage, automatically converts categorical and multi-value categorical features to sparse representation.
- Supports training for both explicit and implicit datasets, as well as negative sampling on implicit data.
- Provides end-to-end workflow, i.e. data handling / preprocessing -> model training -> evaluate -> save/load -> serving.
- Supports cold-start prediction and recommendation.
- Supports dynamic feature and sequence recommendation.
- Provides unified and friendly API for all algorithms. 
- Easy to retrain model with new users/items from new data.

## Usage

#### *pure collaborative-filtering example* :

```python
import pandas as pd
from recora.data import random_split, DatasetPure
from recora.algorithms import BPR  # pure data, algorithm BPR
from recora.evaluation import evaluate

data = pd.read_csv("examples/sample_data/sample_movielens_rating.dat", sep="::",
                   names=["user", "item", "label", "time"])

# split whole data into three folds for training, evaluating and testing
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)
print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %

bpr = BPR(
    task="ranking",
    data_info=data_info,
    loss_type="bpr",
    embed_size=16,
    n_epochs=3,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
)
# monitor metrics on eval data during training
bpr.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    eval_data=eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)

# do final evaluation on test data
evaluate(
    model=bpr,
    data=test_data,
    neg_sampling=True,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)

# predict preference of user 2211 to item 110
bpr.predict(user=2211, item=110)
# recommend 7 items for user 2211
bpr.recommend_user(user=2211, n_rec=7)

# cold-start prediction
bpr.predict(user="ccc", item="not item", cold_start="average")
# cold-start recommendation
bpr.recommend_user(user="are we good?", n_rec=7, cold_start="popular")
```

#### *include features example* :

```python
import numpy as np
import pandas as pd
from recora.data import split_by_ratio_chrono, DatasetFeat
from recora.algorithms import YouTubeRanking  # feat data, algorithm YouTubeRanking

data = pd.read_csv("examples/sample_data/sample_movielens_merged.csv", sep=",", header=0)
# split into train and test data based on time
train_data, test_data = split_by_ratio_chrono(data, test_size=0.2)

# specify complete columns information
sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
dense_col = ["age"]
user_col = ["sex", "age", "occupation"]
item_col = ["genre1", "genre2", "genre3"]

train_data, data_info = DatasetFeat.build_trainset(
    train_data, user_col, item_col, sparse_col, dense_col
)
test_data = DatasetFeat.build_testset(test_data)
print(data_info)  # n_users: 5962, n_items: 3226, data sparsity: 0.4185 %

ytb_ranking = YouTubeRanking(
    task="ranking",
    data_info=data_info,
    embed_size=16,
    n_epochs=3,
    lr=1e-4,
    batch_size=512,
    use_bn=True,
    hidden_units=(128, 64, 32),
)
ytb_ranking.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    shuffle=True,
    eval_data=test_data,
    metrics=["loss", "roc_auc", "precision", "recall", "map", "ndcg"],
)

# predict preference of user 2211 to item 110
ytb_ranking.predict(user=2211, item=110)
# recommend 7 items for user 2211
ytb_ranking.recommend_user(user=2211, n_rec=7)

# cold-start prediction
ytb_ranking.predict(user="ccc", item="not item", cold_start="average")
# cold-start recommendation
ytb_ranking.recommend_user(user="are we good?", n_rec=7, cold_start="popular")
```

## Data Format

JUST normal data format, each line represents a sample. One thing is important, the model assumes that `user`, `item`, and `label` column index are 0, 1, and 2, respectively. You may wish to change the column order if that's not the case. Take for Example, the `movielens-1m` dataset:

> 1::1193::5::978300760  
>
> 1::661::3::978302109  
>
> 1::914::3::978301968  
>
> 1::3408::4::978300275

Besides, if you want to use some other meta features (e.g., age, sex, category etc.),  you need to tell the model which columns are [`sparse_col`, `dense_col`, `user_col`, `item_col`], which means all features must be in a same table. See above `YouTubeRanking` for example.

**Also note that your data should not contain missing values.**

## Documentation

The tutorials and API documentation are hosted on [recora.readthedocs.io](https://recora.readthedocs.io/en/latest/).

The example scripts are under [examples/](https://github.com/samousavizade/MyRec/tree/master/examples) folder.

## Installation & Dependencies

From pypi :  

```shell
$ pip install -U recora
```

Build from source:

```shell
$ git clone https://github.com/samousavizade/MyRec.git
$ cd MyRec
$ pip install .
```

#### Basic Dependencies for `[recora](https://github.com/samousavizade/MyRec/tree/master/recora)`:

- Python >= 3.6
- TensorFlow >= 1.15, < 2.16
- Numpy >= 1.19.5
- Pandas >= 1.0.0
- Scipy >= 1.2.1, < 1.13.0
- scikit-learn >= 0.20.0
- tqdm
- [nmslib](https://github.com/nmslib/nmslib) (optional, used in approximate similarity searching. See [Embedding](https://recora.readthedocs.io/en/latest/user_guide/embedding.html))
- Cython >= 0.29.0, < 3 (optional, for building from source)

If you are using Python 3.6, you also need to install [dataclasses](https://github.com/ericvsmith/dataclasses), which was first introduced in Python 3.7.

Recora has been tested under TensorFlow 1.15, 2.6, 2.10 and 2.12. If you encounter any problem during running, feel free to open an issue.

**Tensorflow [2.16](https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1) starts using Keras 3.0, so tf1 syntax is no longer supported.** Now the supported version is 1.15 - 2.15.

**Known issue**:

- Sometimes one may encounter errors like `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`. In this case try upgrading numpy, and version 1.22.0 or higher is probably a safe option.
- When saving a TensorFlow model for serving, you might encounter the error message: `Fatal Python error: Segmentation fault (core dumped)`.
This issue is most likely related to the `protobuf` library, so you should follow the official recommended [version](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py#L98) 
based on your local tensorflow version. In general, it's advisable to use protobuf < 4.24.0.

The table below shows some compatible version combinations: 


| Python | Numpy                  | TensorFlow      | OS                    |
| ------ | ---------------------- | --------------- | --------------------- |
| 3.6    | 1.19.5                 | 1.15, 2.5       | linux, windows, macos |
| 3.7    | 1.20.3, 1.21.6         | 1.15, 2.6, 2.10 | linux, windows, macos |
| 3.8    | 1.22.4, 1.23.4         | 2.6, 2.10, 2.12 | linux, windows, macos |
| 3.9    | 1.22.4, 1.23.4         | 2.6, 2.10, 2.12 | linux, windows, macos |
| 3.10   | 1.22.4, 1.23.4, 1.24.2 | 2.10, 2.12      | linux, windows, macos |
| 3.11   | 1.23.4, 1.24.2         | 2.12            | linux, windows, macos |


#### Optional Dependencies for `[libserving](https://github.com/samousavizade/MyRec/tree/master/libserving)`:

- Python >= 3.7
- sanic >= 22.3
- requests
- aiohttp
- pydantic
- [ujson](https://github.com/ultrajson/ultrajson)
- [redis](https://redis.io/)
- [redis-py](https://github.com/andymccurdy/redis-py) >= 4.2.0
- [faiss](https://github.com/facebookresearch/faiss) >= 1.5.2
- [TensorFlow Serving](https://github.com/tensorflow/serving) == 2.8.2

## Docker

One can also use the library in a docker container without installing dependencies, see [Docker](https://github.com/samousavizade/MyRec/tree/master/docker).

## References


| Algorithm         | Category[1](#fn1) | Backend             | Sequence[2](#fn2)  | Graph[3](#fn3) | Embedding[4](#fn4) | Paper                                                                                                                                                                                                                                                                                                                                                  |
| ----------------- | ----------------- | ------------------- | ------------------ | -------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| userCF / itemCF   | pure              | Cython, Rust        |                    |                |                    | [Item-Based Collaborative Filtering](http://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf)                                                                                                                                                                                                                                                          |
| SVD               | pure              | TensorFlow1         |                    |                | :heavy_check_mark: | [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)                                                                                                                                                                                                                                            |
| SVD++             | pure              | TensorFlow1         |                    |                | :heavy_check_mark: | [Factorization Meets the Neighborhood](https://dl.acm.org/citation.cfm?id=1401944)                                                                                                                                                                                                                                                                     |
| ALS               | pure              | Cython              |                    |                | :heavy_check_mark: | 1. [Matrix Completion via Alternating Least Square(ALS)](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf) 2. [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) 3. [Conjugate Gradient for Implicit Feedback](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf) |
| NCF               | pure              | TensorFlow1         |                    |                |                    | [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)                                                                                                                                                                                                                                                                                 |
| BPR               | pure              | Cython, TensorFlow1 |                    |                | :heavy_check_mark: | [Bayesian Personalized Ranking](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)                                                                                                                                                                                                                                                                 |
| LightGCN          | pure              | TensorFlow1         |                    | :heavy_check_mark: | :heavy_check_mark: | [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/pdf/2002.02126.pdf)                                                                                                                                                                                                                                |
| NGCF              | pure              | TensorFlow1         |                    | :heavy_check_mark: | :heavy_check_mark: | [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf)                                                                                                                                                                                                                                                                          |
| Wide & Deep       | feat              | TensorFlow1         |                    |                |                    | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                                                                                                                                                                                                                   |
| FM                | feat              | TensorFlow1         |                    |                |                    | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)                                                                                                                                                                                                                                                                   |
| DeepFM            | feat              | TensorFlow1         |                    |                |                    | [DeepFM](https://arxiv.org/pdf/1703.04247.pdf)                                                                                                                                                                                                                                                                                                         |
| YouTubeRetrieval  | feat              | TensorFlow1         | :heavy_check_mark: |                | :heavy_check_mark: | [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)                                                                                                                                                                                                       |
| GraphSage         | feat              | TensorFlow1         | :heavy_check_mark: |                | :heavy_check_mark: | [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)                                                                                                                                                                                                                                                              |
| PinSage           | feat              | TensorFlow1         | :heavy_check_mark: |                | :heavy_check_mark: | [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01973.pdf)                                                                                                                                                                                                                                          |
| YouTubeRanking    | feat              | TensorFlow1         | :heavy_check_mark: |                |                    | [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)                                                                                                                                                                                                       |
| AutoInt           | feat              | TensorFlow1         |                    |                |                    | [AutoInt](https://arxiv.org/pdf/1810.11921.pdf)                                                                                                                                                                                                                                                                                                        |
| DIN               | feat              | TensorFlow1         | :heavy_check_mark: |                |                    | [Deep Interest Network](https://arxiv.org/pdf/1706.06978.pdf)                                                                                                                                                                                                                                                                                          |
| RNN4Rec / GRU4Rec | pure              | TensorFlow1         | :heavy_check_mark: |                | :heavy_check_mark: | [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939.pdf)                                                                                                                                                                                                                                                   |
| Caser             | pure              | TensorFlow1         | :heavy_check_mark: |                | :heavy_check_mark: | [Personalized Top-N Sequential Recommendation via Convolutional](https://arxiv.org/pdf/1809.07426.pdf)                                                                                                                                                                                                                                                 |
| WaveNet           | pure              | TensorFlow1         | :heavy_check_mark: |                | :heavy_check_mark: | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499.pdf)                                                                                                                                                                                                                                                                      |
| TwoTower          | feat              | TensorFlow1         |                    |                | :heavy_check_mark: | 1. [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/6c8a86c981a62b0126a11896b7f6ae0dae4c3566.pdf) 2. [Self-supervised Learning for Large-scale Item](https://arxiv.org/pdf/2007.12865.pdf)                                                                         |
| Transformer       | feat              | TensorFlow1         | :heavy_check_mark: |                |                    | 1. [BST](https://arxiv.org/pdf/1905.06874.pdf) 2. [Transformers4Rec](https://dl.acm.org/doi/10.1145/3460231.3474255) 3. [RMSNorm](https://arxiv.org/pdf/1910.07467.pdf)                                                                                                                                                                                |
| SIM               | feat              | TensorFlow1         | :heavy_check_mark: |                |                    | [SIM](https://arxiv.org/pdf/2006.05639.pdf)                                                                                                                                                                                                                                                                                                            |
| Swing             | pure              | Rust                |                    |                |                    | [Swing](https://arxiv.org/pdf/2010.05525)                                                                                                                                                                                                                                                                                                              |


> [1] **Category**: `pure` means collaborative-filtering algorithms which only use behavior data, `feat` means other side-features can be included. [↩](#ref1)
>
> [2] **Sequence**: Algorithms that leverage user behavior sequence. [↩](#ref2)
>
> [3] **Graph**: Algorithms that leverage graph information, including Graph Embedding (GE) and Graph Neural Network (GNN) . [↩](#ref3)
>
> [4] **Embedding**: Algorithms that can generate final user and item embeddings. [↩](#ref4)

### Powered by

[JetBrains Logo](https://www.jetbrains.com/community/opensource/#support)
