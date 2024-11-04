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
