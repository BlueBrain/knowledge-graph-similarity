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

from abc import ABC

from similarity_tools.helpers.bucket_configuration import Deployment
from similarity_tools.helpers.constants import SRC_DATA_DIR, DST_DATA_DIR


class ModelData(ABC):
    src_data_dir: str
    dst_data_dir: str
    org: str
    project: str
    deployment: Deployment

    def __init__(
            self, org: str, project: str, deployment: Deployment,
            src_data_dir=None, dst_data_dir=None
    ):
        self.src_data_dir = src_data_dir if src_data_dir is not None else SRC_DATA_DIR
        self.dst_data_dir = dst_data_dir if dst_data_dir is not None else DST_DATA_DIR
        self.deployment = deployment
        self.org = org
        self.project = project
