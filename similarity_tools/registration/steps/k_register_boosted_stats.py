from kgforge.specializations.mappings import DictionaryMapping

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from inference_tools.similarity.formula import Formula
from similarity_tools.helpers.elastic import ElasticSearch
from similarity_tools.helpers.constants import STATISTIC_MAPPING_PATH
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.helper_functions.stat import compute_statistics, \
    register_stats
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.step import Step


def register_boosted_statistics(
    joint_bc: NexusBucketConfiguration,
    similarity_aggregated_view_id: str,
    boosting_aggregated_view_id: str,
    derivation_type: str,
    stats_tag: str,
    score_formula: Formula
) -> str:

    logger.info("1. Retrieving boosting factors")

    forge_joint_boosting_aggregated_view = joint_bc.copy_with_views(
        elastic_search_view=boosting_aggregated_view_id
    ).allocate_forge_session()

    boosting_data = ElasticSearch.get_all_documents(forge_joint_boosting_aggregated_view)
    boosting_data = dict((b.derivation.entity.id, b.value) for b in boosting_data)

    logger.info("2. Computing boosted statistics")

    forge_joint_similarity_aggregated_view = joint_bc.copy_with_views(
        elastic_search_view=similarity_aggregated_view_id
    ).allocate_forge_session()

    stats = compute_statistics(
        forge=forge_joint_similarity_aggregated_view, score_formula=score_formula,
        boosting=boosting_data, derivation_type=derivation_type
    )
    mapping = DictionaryMapping.load(STATISTIC_MAPPING_PATH)

    logger.info("3. Registering statistics")

    stats_id = register_stats(
        forge=forge_joint_similarity_aggregated_view,
        aggregated_similarity_view_id=similarity_aggregated_view_id,
        stats=stats, formula=score_formula, stats_tag=stats_tag,
        boosted=True, mapping=mapping
    )

    logger.info(f">  ElasticSearch Statistics, boosted: {stats_id}")

    return stats_id


step_11 = ModelRegistrationStep(
    function_call=register_boosted_statistics,
    step=Step.REGISTER_BOOSTED_STATS,
    log_message="Registering boosted statistics"
)
