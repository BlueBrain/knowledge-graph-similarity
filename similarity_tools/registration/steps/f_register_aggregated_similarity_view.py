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

from typing import Optional, List, Tuple

from inference_tools.nexus_utils.delta_utils import DeltaException
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.helper_functions.view import create_aggregated_view
from similarity_tools.registration.registration_exception import RegistrationException
from similarity_tools.registration.step import Step
from similarity_tools.helpers.utils import create_id_with_config


def create_aggregated_similarity_view(
    joint_bc: NexusBucketConfiguration,
    to_aggregate: List[Tuple[NexusBucketConfiguration, str]],
) -> str:

    aggregated_view_id = create_id_with_config(joint_bc, is_view=True)

    try:
        agg_view_id = create_aggregated_view(
            bucket_configuration=joint_bc,
            projects_to_aggregate=to_aggregate,
            aggregated_view_id=aggregated_view_id
        )
        logger.info(f">  Aggregated Similarity view id: {aggregated_view_id}")
        return agg_view_id

    except DeltaException as e:
        raise RegistrationException(f"Could not create aggregated similarity view: {e.body}")


step_6 = ModelRegistrationStep(
    function_call=create_aggregated_similarity_view,
    step=Step.REGISTER_SIMILARITY_VIEW,
    log_message="Creating aggregated similarity view"
)
