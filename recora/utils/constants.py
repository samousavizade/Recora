from enum import Enum, unique


class StrEnum(str, Enum):
    @classmethod
    def contains(cls, x):
        return x in cls.__members__.values()  # cls._member_names_


@unique
class FeatModels(StrEnum):
    WIDEDEEP = "WideDeep"
    FM = "FM"
    DEEPFM = "DeepFM"
    YOUTUBERETRIEVAL = "YouTubeRetrieval"
    YOUTUBERANKING = "YouTubeRanking"
    AUTOINT = "AutoInt"
    DIN = "DIN"
    TWOTOWER = "TwoTower"
    TRANSFORMER = "Transformer"
    SIM = "SIM"


@unique
class SequenceModels(StrEnum):
    YOUTUBERETRIEVAL = "YouTubeRetrieval"
    YOUTUBERANKING = "YouTubeRanking"
    DIN = "DIN"
    RNN4REC = "RNN4Rec"
    CASER = "Caser"
    WAVENET = "WaveNet"
    TRANSFORMER = "Transformer"
    SIM = "SIM"


@unique
class TfTrainModels(StrEnum):
    SVD = "SVD"
    SVDPP = "SVDpp"
    NCF = "NCF"
    BPR = "BPR"
    LIGHTGCN = "LightGCN"
    NGCF = "NGCF"
    WIDEDEEP = "WideDeep"
    FM = "FM"
    DEEPFM = "DeepFM"
    YOUTUBERETRIEVAL = "YouTubeRetrieval"
    YOUTUBERANKING = "YouTubeRanking"
    AUTOINT = "AutoInt"
    DIN = "DIN"
    RNN4REC = "RNN4Rec"
    CASER = "Caser"
    WAVENET = "WaveNet"
    TWOTOWER = "TwoTower"
    TRANSFORMER = "Transformer"
    SIM = "SIM"


@unique
class EmbeddingModels(StrEnum):
    SVD = "SVD"
    SVDPP = "SVDpp"
    ALS = "ALS"
    BPR = "BPR"
    LIGHTGCN = "LightGCN"
    NGCF = "NGCF"
    YOUTUBERETRIEVAL = "YouTubeRetrieval"
    RNN4REC = "RNN4Rec"
    CASER = "Caser"
    WAVENET = "WaveNet"
    TWOTOWER = "TwoTower"


@unique
class UserEmbedModels(StrEnum):
    """Models can only generate user embeddings dynamically."""

    YOUTUBERETRIEVAL = "YouTubeRetrieval"
    RNN4REC = "RNN4Rec"
    CASER = "Caser"
    WAVENET = "WaveNet"
