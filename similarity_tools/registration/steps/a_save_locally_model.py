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
import json
from typing import Optional, Union, Dict, List

from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.helpers.logger import logger

from similarity_tools.data_classes.model import Model
from similarity_tools.data_classes.model_description import ModelDescription
from similarity_tools.data_classes.model_data import ModelData

from similarity_tools.helpers.constants import DST_DATA_DIR, PIPELINE_SUBDIRECTORY
from similarity_tools.registration.step import Step

from bluegraph.downstream import EmbeddingPipeline


def save_locally_model(
        model_description: ModelDescription,
        model_data: Optional[ModelData],
        pipeline: Optional[Union[EmbeddingPipeline, Dict[str, List]]] = None
):
    if not pipeline:
        logger.info("1. Initializing model")

        model_instance: Model = model_description.model(model_data)

        logger.info("2. Running model")

        pipeline: Union[EmbeddingPipeline, Dict[str, List]] = model_instance.run()

    filename = f"{model_description.filename}_{model_data.org}_{model_data.project}_{model_data.deployment.name.lower()}"

    if pipeline is None:
        logger.warning(
            f"No embeddings were computed for model {model_description.name} and bucket "
            f"{model_data.org}/{model_data.project}"
        )
        return None

    pipeline_directory = os.path.join(DST_DATA_DIR, PIPELINE_SUBDIRECTORY)
    save_path = os.path.join(pipeline_directory, filename)

    logger.info(f"3. Saving model {model_description.name} to {save_path}")

    os.makedirs(os.path.dirname(pipeline_directory), exist_ok=True)

    if isinstance(pipeline, EmbeddingPipeline):
        pipeline.save(save_path, compress=True)
    else:
        with open(f"{save_path}.json", "w") as outfile:
            json.dump(pipeline, outfile)

    return save_path


step_1 = ModelRegistrationStep(
    function_call=save_locally_model,
    step=Step.SAVE_MODEL,
    log_message="Running and downloading locally"
)
