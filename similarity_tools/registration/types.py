# This file is part of knowledge-graph-similarity.
# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class Types(Enum):
    SIMILARITY_BOOSTING_FACTOR = "SimilarityBoostingFactor"
    EMBEDDING_MODEL = "EmbeddingModel"
    EMBEDDING_MODEL_DATA_CATALOG = "EmbeddingModelDataCatalog"
    EMBEDDING = "Embedding"
    ES_VIEW_STATS = "ElasticSearchViewStatistics"
