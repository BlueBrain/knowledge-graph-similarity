from enum import Enum


class Step(Enum):
    SAVE_MODEL = 1
    REGISTER_MODEL = 2
    REGISTER_EMBEDDING_MODEL_CATALOG = 3
    REGISTER_EMBEDDINGS = 4
    REGISTER_SIMILARITY_VIEW = 5
    REGISTER_AGGREGATED_SIMILARITY_VIEW = 6
    REGISTER_NON_BOOSTED_STATS = 7
    REGISTER_BOOSTING_FACTORS = 8
    REGISTER_BOOSTING_VIEW = 9
    REGISTER_AGGREGATED_BOOSTING_VIEW = 10
    REGISTER_BOOSTED_STATS = 11
    REGISTER_STATS_VIEW = 12
    # PLOT_2D_EMBEDDINGS =
