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

from typing import Tuple

from inference_tools.nexus_utils.delta_utils import DeltaException
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.mappings.es_mappings import get_es_view_mappings, get_es_view_binary_mappings
from similarity_tools.registration.helper_functions.view import view_create
from similarity_tools.registration.registration_exception import RegistrationException
from similarity_tools.registration.step import Step
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import create_id_with_config


def create_similarity_view(
        bucket_configuration: NexusBucketConfiguration,
        resource_tag: str,
        vector_dimension: int
) -> str:

    """
    Creates a similarity view in the bucket tied to the bucket
    configuration, pointing to Embedding.s in that same bucket and tagged by the resource tag.

    @param bucket_configuration: a bucket configuration pointing to the bucket where the view will
    be created
    @type bucket_configuration: NexusBucketConfiguration
    # @param view_id: the id of the similarity view to create
    # @type view_id: str
    @param resource_tag: the tag the embeddings targeted must have
    @type resource_tag: str
    @param vector_dimension: the dimension of the vector held in the embedding, for ES indexing
    @type vector_dimension: int
    @return: the similarity view id
    @rtype: str
    """

    view_id = create_id_with_config(bucket_configuration, is_view=True)

    try:
        if vector_dimension >= 4096:
            mapping = get_es_view_binary_mappings()
        else:
            mapping = get_es_view_mappings(vector_dimension)
        view_create(
            mapping=mapping,
            view_id=view_id,
            resource_types=[f"https://neuroshapes.org/{Types.EMBEDDING.value}"],
            bucket_configuration=bucket_configuration,
            resource_tag=resource_tag
        )
        logger.info(f">  Similarity view id: {view_id}")
        return view_id

    except DeltaException as e:
        raise RegistrationException(f"Could not create similarity view: {e}")


step_5 = ModelRegistrationStep(
    function_call=create_similarity_view,
    step=Step.REGISTER_SIMILARITY_VIEW,
    log_message="Creating similarity view"
)
