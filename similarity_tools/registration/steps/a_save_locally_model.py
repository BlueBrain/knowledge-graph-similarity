import os
from typing import Optional

from similarity_tools.helpers.bucket_configuration import Deployment
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.helpers.logger import logger

from similarity_tools.data_classes.model import Model
from similarity_tools.data_classes.model_description import ModelDescription
from similarity_tools.data_classes.model_data import ModelData

from similarity_tools.helpers.constants import DST_DATA_DIR, PIPELINE_SUBDIRECTORY
from similarity_tools.registration.step import Step


def save_locally_model(
        model_description: ModelDescription,
        model_data: Optional[ModelData],
):
    logger.info("1. Initializing model")

    model_instance: Model = model_description.model(model_data)

    filename = f"{model_description.init_filename}_{model_data.org}_{model_data.project}_{model_data.deployment.name.lower()}"

    logger.info("2. Running model")

    pipeline = model_instance.run()

    if pipeline is None:
        logger.warning(f"No embeddings were computed for model {model_description.name} and bucket "
                       f"{model_data.org}/{model_data.project}")
        return None

    pipeline_directory = os.path.join(DST_DATA_DIR, PIPELINE_SUBDIRECTORY)
    save_path = os.path.join(pipeline_directory, filename)

    logger.info(f"3. Saving model {model_description.name} to {save_path}")

    os.makedirs(os.path.dirname(pipeline_directory), exist_ok=True)
    pipeline.save(save_path, compress=True)

    return save_path


step_1 = ModelRegistrationStep(
    function_call=save_locally_model,
    step=Step.SAVE_MODEL,
    log_message="Running and downloading locally"
)
