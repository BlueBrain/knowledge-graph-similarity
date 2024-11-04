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

from kgforge.core import Resource

from inference_tools.datatypes.similarity.statistic import Statistic
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from inference_tools.similarity.formula import Formula
from similarity_tools.helpers.constants import BOOSTING_FACTOR_MAPPING_PATH
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.helper_functions.boosting_factor import (
    compute_boosting_factors,
    register_boosting_factors
)
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import SimilarityToolsException
from similarity_tools.registration.step import Step


def register_boosting_data(
        joint_bc: NexusBucketConfiguration,
        bucket_bc: NexusBucketConfiguration,
        aggregated_similarity_view_id: str,
        non_boosted_stats_id: str,
        score_formula: Formula,
        boosting_tag: str
) -> str:
    """

    @param joint_bc:
    @type joint_bc: NexusBucketConfiguration
    @param bucket_bc:
    @type bucket_bc: NexusBucketConfiguration
    @param aggregated_similarity_view_id:
    @type aggregated_similarity_view_id: str
    @param non_boosted_stats_id:
    @type non_boosted_stats_id: str
    @param score_formula:
    @type score_formula: Formula
    @param boosting_tag:
    @type boosting_tag: str
    @return: the resource tag the boosting factors were tagged with
    @rtype:
    """

    logger.info("2. Retrieving non-boosted statistics")

    forge_joint = joint_bc.allocate_forge_session()
    stats: Resource = forge_joint.retrieve(non_boosted_stats_id)
    stats: Statistic = Statistic.from_json(forge_joint.as_json(stats))

    if stats is None:
        raise SimilarityToolsException(
            "Could not retrieve non-boosted statistics, first run step register non boosted stats"
        )

    logger.info("3. Computing boosting factors")

    forge_joint_aggregated_similarity_view = joint_bc.copy_with_views(
        elastic_search_view=aggregated_similarity_view_id
    ).allocate_forge_session()

    boosting = compute_boosting_factors(
        forge=forge_joint_aggregated_similarity_view,
        stats=stats,
        formula=score_formula
    )

    logger.info("4. Registering boosting factors")

    forge_bucket = bucket_bc.allocate_forge_session()

    register_boosting_factors(
        forge=forge_bucket,
        view_id=aggregated_similarity_view_id,
        boosting_factors=boosting,
        formula=score_formula,
        boosting_tag=boosting_tag,
        mapping_path=BOOSTING_FACTOR_MAPPING_PATH
    )

    return boosting_tag


step_8 = ModelRegistrationStep(
    function_call=register_boosting_data,
    step=Step.REGISTER_BOOSTING_FACTORS,
    log_message="Registering boosting data"
)
