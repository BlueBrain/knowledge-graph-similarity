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

from inference_tools.nexus_utils.delta_utils import DeltaException
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.helper_functions.view import view_create
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.mappings.es_mappings import BOOSTING_VIEW_MAPPING
from similarity_tools.registration.registration_exception import RegistrationException
from similarity_tools.registration.step import Step
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import create_id_with_config


def create_boosting_view(
    bucket_configuration: NexusBucketConfiguration,
    boosting_tag: str
) -> str:

    boosting_view_id = create_id_with_config(bucket_configuration, is_view=True)

    try:
        view_create(
            mapping=BOOSTING_VIEW_MAPPING,
            view_id=boosting_view_id,
            resource_types=[f"https://neuroshapes.org/{Types.SIMILARITY_BOOSTING_FACTOR.value}"],
            bucket_configuration=bucket_configuration,
            resource_tag=boosting_tag
        )
        logger.info(f">  Boosting view id: {boosting_view_id}")
        return boosting_view_id
    except DeltaException as e:
        raise RegistrationException(f"Could not create boosting view : {e}")


step_9 = ModelRegistrationStep(
    function_call=create_boosting_view,
    step=Step.REGISTER_BOOSTING_VIEW,
    log_message="Creating boosting view"
)
