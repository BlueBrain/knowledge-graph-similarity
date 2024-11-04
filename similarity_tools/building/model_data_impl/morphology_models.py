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

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration

from similarity_tools.data_classes.model_data import ModelData
from similarity_tools.helpers.logger import logger


class ModelDataMorphologyModels(ModelData):

    def __init__(
            self, bucket_configuration: NexusBucketConfiguration,
            src_data_dir=None, dst_data_dir=None,
    ):
        super().__init__(
            src_data_dir=src_data_dir, dst_data_dir=dst_data_dir,
            deployment=bucket_configuration.deployment,
            org=bucket_configuration.organisation, project=bucket_configuration.project
        )

        logger.info(">  Load morphological models from Nexus")

        self.forge = bucket_configuration.allocate_forge_session()

        self.data = self.forge.search({"type": "CanonicalMorphologyModel"}, limit=1500)
