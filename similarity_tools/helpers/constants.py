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

from similarity_tools.helpers.utils import get_path

SRC_DATA_DIR = get_path("../data_2/")
DST_DATA_DIR = get_path("../data_2/")
PIPELINE_SUBDIRECTORY = "pipelines/"
PLOT_SUBDIRECTORY = "plots/"
TSNE_SUBDIRECTORY = "tsne/"
EMBEDDING_MAPPING_PATH = get_path("./registration/mappings/embedding.hjson")
BOOSTING_FACTOR_MAPPING_PATH = get_path("./registration/mappings/boosting_factor.hjson")
STATISTIC_MAPPING_PATH = get_path("./registration/mappings/statistic.hjson")
