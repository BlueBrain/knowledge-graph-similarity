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
    data_bc: NexusBucketConfiguration,
    push_bc: NexusBucketConfiguration,
    model_description: Optional[ModelDescription] = None,
    model_bc: Optional[NexusBucketConfiguration] = None,
    model_path: Optional[str] = None,
    model: Optional[Resource] = None,
    resource_id_rev_list: Optional[List[Tuple[str, str]]] = None,
    embedding_tag_transformer: Optional[Callable[[str], str]] = None,
    bluegraph: bool = False,
    tag: Optional[str] = None
) -> Tuple[str, int]:
    """

    @param data_bc: a bucket configuration pointing to where the data being embedded is located
    @type data_bc: NexusBucketConfiguration
    @param model_bc: a bucket configuration pointing to where model is located
    @type model_bc: NexusBucketConfiguration
    @param push_bc: a bucket configuration pointing to where the embeddings will be pushed
    @type push_bc: NexusBucketConfiguration
    @param model_description:
    @type model_description: Optional[ModelDescription]
    @param model_path: where the model pipeline is located, if ever a local copy is used instead of a
    model resource
    @type model_path Optional[str]
    @param model:
    @type model: Optional[Resource]
    @param resource_id_rev_list: if ever only a subset of the embeddings should be registered
    @type resource_id_rev_list:
    @param tag: a tag to apply to the embeddings
    @type tag: Optional[str]
    @param embedding_tag_transformer: by default, the embedding tag will be the provided tag and if none is provided,
    a concatenation of the model uuid and its rev. If this parameter is specified, it transforms this tag into a
    user specified tag (can be an entirely new tag or a transformation of the default one). The
    default one is used if no transformation is provided
    @type embedding_tag_transformer Optional[Callable[[str], str]]
    @param bluegraph: whether the embedding process was done using the bluegraph library or not
    (and therefore whether it should be added to the derivation and generation of embeddings)
    @type bluegraph: bool

    # @param entity_type: the type of the entities being embedded
    # @type entity_type: str

    @return: the tag applied to the embeddings, and the vector dimension of the embeddings
    @rtype: Tuple[str, int]
    """

    if model_path is None \
            and (model_description is None and model_bc is None)\
            and model is None:

        raise Exception(
            "No way to retrieve the model: No model provided,"
            " no model path provided, and no model description + model bucket configuration provided"
        )

    if model_path is None:
        forge_model = model_bc.allocate_forge_session()

        if model is None:
            logger.info("1. Fetching model")

            model = fetch_model(
                forge_model, model_name=model_description.name, model_rev=model_description.model_rev
            )
            if not model:
                raise SimilarityToolsException(f"Error retrieving model {model_description.name}")
        else:
            logger.info("1. Model provided")

        model_revision = model._store_metadata['_rev']
        model_id = model.id
    else:
        logger.info("1. Loading model from provided path, the embeddings will not be tagged")
        forge_model = None
        model_id = "file"  # TODO not good because forge tried to expand it as an id, and further _search checks will not pass
        model_revision = None

    pipeline_directory = os.path.join(DST_DATA_DIR, PIPELINE_SUBDIRECTORY)

    logger.info("2. Loading model in memory")

    model_revision, model_tag, pipeline = load_embedding_model(
        forge=forge_model,
        model_revision=model_revision,
        download_dir=pipeline_directory,
        path=model_path,
        model_id=model_id,
        tag=tag
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

    forge_push = push_bc.allocate_forge_session() if push_bc != model_bc else forge_model
    forge_data = data_bc.allocate_forge_session() if data_bc != model_bc else forge_model

    embedding_tag, vector_dimension = register_embeddings(
        forge_data=forge_data,
        forge_push=forge_push,
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
