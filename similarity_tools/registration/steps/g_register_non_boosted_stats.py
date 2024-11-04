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

from kgforge.specializations.mappings import DictionaryMapping

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from inference_tools.similarity.formula import Formula
from similarity_tools.helpers.constants import STATISTIC_MAPPING_PATH
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.helper_functions.stat import compute_statistics, \
    register_stats
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.step import Step


def register_non_boosted_statistics(
    joint_bc: NexusBucketConfiguration,
    similarity_aggregated_view_id: str,
    derivation_type: str,
    stats_tag: str,
    score_formula: Formula
) -> str:
    logger.info("1. Computing non-boosted statistics")

    forge_embeddings = joint_bc.copy_with_views(
        elastic_search_view=similarity_aggregated_view_id
    ).allocate_forge_session()

    stats = compute_statistics(
        forge=forge_embeddings, score_formula=score_formula, boosting=None,
        derivation_type=derivation_type
    )

    mapping = DictionaryMapping.load(STATISTIC_MAPPING_PATH)

    logger.info("2. Registering statistics")

    stats_id = register_stats(
        forge=forge_embeddings,
        aggregated_similarity_view_id=similarity_aggregated_view_id,
        stats=stats,
        formula=score_formula,
        stats_tag=stats_tag,
        boosted=False,
        mapping=mapping
    )

    logger.info(f">  ElasticSearch Statistics, non boosted: {stats_id}")

    return stats_id


step_7 = ModelRegistrationStep(
    function_call=register_non_boosted_statistics,
    step=Step.REGISTER_NON_BOOSTED_STATS,
    log_message="Registering non-boosted statistics"
)
