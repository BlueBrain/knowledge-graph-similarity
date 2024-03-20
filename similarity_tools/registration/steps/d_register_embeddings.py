import os
from importlib.resources import Resource
from typing import List, Tuple, Optional, Callable

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.registration.helper_functions.model import fetch_model
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.helpers.logger import logger

from similarity_tools.data_classes.model_description import ModelDescription

from similarity_tools.registration.helper_functions.embedding import (
    register_embeddings,
    load_embedding_model,
    get_embedding_vectors_from_pipeline
)
from similarity_tools.helpers.constants import DST_DATA_DIR, PIPELINE_SUBDIRECTORY, EMBEDDING_MAPPING_PATH

from similarity_tools.registration.registration_exception import SimilarityToolsException
from similarity_tools.registration.step import Step


def register_model_embeddings(
    model_bc: NexusBucketConfiguration,
    model_description: ModelDescription,
    model: Optional[Resource] = None,
    resource_id_rev_list: Optional[List[Tuple[str, str]]] = None,
    embedding_tag_transformer: Optional[Callable[[str], str]] = None,
    bluegraph: bool = False
) -> Tuple[str, int]:
    """

    @param model_bc:
    @type model_bc: NexusBucketConfiguration
    @param model_description:
    @type model_description: ModelDescription
    # @param entity_type: the type of the entities being embedded
    # @type entity_type: str
    @param model:
    @type model: Optional[Resource]
    @param resource_id_rev_list: if ever only a subset of the embeddings should be registered
    @type resource_id_rev_list:
    @param embedding_tag_transformer: by default, the embedding tag will be a concatenation of
    the model uuid and its rev. If this parameter is specified, it transforms this tag into a
    user specified tag (can be an entirely new tag or a transformation of the default one). The
    default one is used if no transformation is provided
    @type embedding_tag_transformer Optional[Callable[[str], str]]
    @return: the tag applied to the embeddings, and the vector dimension of the embeddings
    @rtype: Tuple[str, int]
    """
    forge_model = model_bc.allocate_forge_session()

    if model is None:
        logger.info("1. Fetching model")
        model = fetch_model(forge_model, model_name=model_description.name)
        if not model:
            raise SimilarityToolsException(f"Error retrieving model {model_description.name}")
    else:
        logger.info("1. Model provided")

    model_id = model.id

    pipeline_directory = os.path.join(DST_DATA_DIR, PIPELINE_SUBDIRECTORY)

    logger.info("2. Loading model in memory")

    model_revision, model_tag, pipeline = load_embedding_model(
        forge_model, model_revision=model_description.model_rev, download_dir=pipeline_directory,
        model_id=model_id
    )

    model_tag = embedding_tag_transformer(model_tag) if embedding_tag_transformer is not None \
        else model_tag

    logger.info("3. Getting embedding vectors from model pipeline")

    missing_list, embedding_dict = get_embedding_vectors_from_pipeline(
        pipeline=pipeline, resource_id_rev_list=resource_id_rev_list
    )

    if resource_id_rev_list is not None:
        logger.info(f">  Number of missing embeddings in the embedding table: {len(missing_list)}")

    logger.info("4. Registering embeddings")

    embedding_tag, vector_dimension = register_embeddings(
        forge=forge_model,
        vectors=embedding_dict,
        model_revision=model_revision,
        model_id=model_id,
        embedding_tag=model_tag,
        mapping_path=EMBEDDING_MAPPING_PATH,
        bluegraph=bluegraph
    )

    logger.info(f">  Embedding tags: {embedding_tag}")
    logger.info(f">  Embedding vector dimension: {vector_dimension}")

    return embedding_tag, vector_dimension


step_4 = ModelRegistrationStep(
    function_call=register_model_embeddings,
    step=Step.REGISTER_EMBEDDINGS,
    log_message="Registering embeddings"
)
