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

import os

from kgforge.core import Resource, KnowledgeGraphForge
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration

from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.helpers.logger import logger
from similarity_tools.data_classes.model_description import ModelDescription
from similarity_tools.registration.helper_functions.model import push_model
from similarity_tools.helpers.constants import DST_DATA_DIR, PIPELINE_SUBDIRECTORY
from similarity_tools.registration.step import Step
from similarity_tools.helpers.utils import raise_error_on_failure


def register_model(
    model_description: ModelDescription,
    model_bc: NexusBucketConfiguration
) -> str:

    pipeline_directory = os.path.join(DST_DATA_DIR, PIPELINE_SUBDIRECTORY)
    load_pipeline_path = os.path.join(pipeline_directory, model_description.filename)

    logger.info(f">  Location: {model_bc}")

    model_resource: Resource = push_model(
        forge=model_bc.allocate_forge_session(),
        model_name=model_description.name,
        description=model_description.description,
        pipeline_path=load_pipeline_path,
        distance_metric=model_description.distance,
        label=model_description.label
    )

    raise_error_on_failure(model_resource)

    logger.info(f">  Model identifier: {model_resource.get_identifier()}")

    return model_resource.get_identifier()


step_2 = ModelRegistrationStep(
    function_call=register_model,
    step=Step.REGISTER_MODEL,
    log_message="Pushing to Nexus"
)
