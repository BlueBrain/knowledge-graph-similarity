from typing import Optional, List, Tuple

from inference_tools.nexus_utils.delta_utils import DeltaException
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.helper_functions.view import create_aggregated_view
from similarity_tools.registration.registration_exception import RegistrationException
from similarity_tools.registration.step import Step
from similarity_tools.helpers.utils import create_id_with_config


def create_aggregated_boosting_view(
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
        logger.info(f">  Aggregated Boosting view id: {aggregated_view_id}")
        return agg_view_id

    except DeltaException as e:
        raise RegistrationException(f"Could not create aggregated boosting view: {e.body}")


step_10 = ModelRegistrationStep(
    function_call=create_aggregated_boosting_view,
    step=Step.REGISTER_AGGREGATED_BOOSTING_VIEW,
    log_message="Creating aggregated boosting view"
)
