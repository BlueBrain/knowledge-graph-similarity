from typing import List, Optional, Tuple, Dict

from kgforge.core import Resource, KnowledgeGraphForge
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.registration.helper_functions.embedding_model_data_catalog import \
    push_embedding_model_data_catalog

from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.step import Step
from similarity_tools.helpers.utils import raise_error_on_failure


def register_embedding_model_catalog(
        model_name: str,
        joint_bc: NexusBucketConfiguration,
        bucket_list_rev: List[Tuple[NexusBucketConfiguration, Optional[int]]],
        target_type: str
) -> str:
    """
    Creates an embedding model catalog with name "Catalog of {model_label} models"
    Its hasPart contains EmbeddingModel.s with name {model_label} located in the buckets
    specified by bucket_list_rev. EmbeddingModel.s are searched in each bucket
    @param model_name:
    @type model_name: str
    @param joint_bc:
    @type joint_bc: NexusBucketConfiguration
    @param bucket_list_rev:
    @type bucket_list_rev: List[Tuple[NexusBucketConfiguration, Optional[int]]
    @param target_type: the type of the entity being embedded by the parts of the catalog
    @type target_type: str
    @return:
    @rtype:
    """

    logger.info(f">  Location: {joint_bc}")

    joint_forge: KnowledgeGraphForge = joint_bc.allocate_forge_session()

    catalog_resource: Resource = push_embedding_model_data_catalog(
        model_name=model_name,
        joint_forge=joint_forge, bucket_list=bucket_list_rev,
        target_type=target_type
    )

    raise_error_on_failure(catalog_resource)

    logger.info(f">  Embedding Model Data Catalog identifier: {catalog_resource.get_identifier()}")

    return catalog_resource.get_identifier()


step_3 = ModelRegistrationStep(
    function_call=register_embedding_model_catalog,
    step=Step.REGISTER_EMBEDDING_MODEL_CATALOG,
    log_message="Register Embedding Model Data Catalog"
)
