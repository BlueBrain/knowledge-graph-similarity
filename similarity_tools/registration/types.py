from enum import Enum


class Types(Enum):
    SIMILARITY_BOOSTING_FACTOR = "SimilarityBoostingFactor"
    EMBEDDING_MODEL = "EmbeddingModel"
    EMBEDDING_MODEL_DATA_CATALOG = "EmbeddingModelDataCatalog"
    EMBEDDING = "Embedding"
    ES_VIEW_STATS = "ElasticSearchViewStatistics"
