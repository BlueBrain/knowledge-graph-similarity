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

from similarity_tools.building.model_data_impl.morphology_models import ModelDataMorphologyModels

from similarity_tools.building.model_impl.tmd_model.tmd_model_with_mm import TMDModelWithMM, \
    UnscaledTMDModelWithMM

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

re_download = False
re_compute = True

deployment = Deployment.PRODUCTION

bucket_models = NexusBucketConfiguration(
    organisation="bbp", project="mmb-point-neuron-framework-model", deployment=deployment
)

test = UnscaledTMDModelWithMM(
    model_data=ModelDataMorphologyModels(bucket_configuration=bucket_models),
    re_compute=re_compute, re_download=re_download
)
print(test.run())
