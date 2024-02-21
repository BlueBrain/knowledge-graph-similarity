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
    load_pipeline_path = os.path.join(pipeline_directory, f"{model_description.filename}.zip")

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

