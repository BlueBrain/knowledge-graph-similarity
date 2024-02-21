import os
from typing import List, Dict, Optional, Tuple, Callable

from bluegraph.core import GraphElementEmbedder
from bluegraph.downstream import EmbeddingPipeline
from kgforge.core import KnowledgeGraphForge, Resource
from kgforge.specializations.mappings import DictionaryMapping

from similarity_tools.registration.helper_functions.common import _persist
from similarity_tools.registration.helper_functions.model import _software_agent_bluegraph
from similarity_tools.registration.helper_functions.software_agents import \
    _software_agent_similarity_tools
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import encode_id_rev, get_model_tag, parse_id_rev, create_id, \
    create_id_with_forge


def load_embedding_model(
        forge: KnowledgeGraphForge,
        model_id: str,
        model_revision: Optional[int] = None,
        download_dir: str = ".",
) -> Tuple[int, str, EmbeddingPipeline]:
    """
    Load embedding model embedding pipeline zip file into memory
    @param forge: a forge instance
    @type forge: KnowledgeGraphForge
    @param model_id: the id of the model
    @type model_id: str
    @param model_revision: the revision of the model
    @type model_revision: Optional[int]
    @param download_dir: where to download the model distribution locally
    @type download_dir: str
    @return: the specified model revision if it was specified else the latest,
    a tag made of the concatenation of the model uuid and rev, and the model embedding pipeline
    @rtype: Tuple[int, str, EmbeddingPipeline]
    """

    retrieval_str = f"{model_id}{'?rev='}{model_revision}" \
        if model_revision is not None else model_id

    model = forge.retrieve(retrieval_str)

    # If revision is not provided by the user, fetch the latest
    model_revision = model_revision if model_revision is not None else model._store_metadata._rev

    model_tag = get_model_tag(model_id, model_revision)

    forge.download(
        model, "distribution.contentUrl", download_dir, overwrite=True)

    path = os.path.join(download_dir, model.distribution.name)

    pipeline = EmbeddingPipeline.load(
        path=path, embedder_interface=GraphElementEmbedder, embedder_ext="zip"
    )

    return model_revision, model_tag, pipeline


def get_embedding_vectors_from_pipeline(
        pipeline: EmbeddingPipeline,
        resource_id_rev_list: Optional[List[Tuple[str, int]]] = None
) -> Tuple[
    List[Tuple[str, int]],  # missing embeddings
    Dict[Tuple[str, int], List[float]]  # embeddings found
]:
    """
    Get embedding vectors from an Embedding Pipeline

    @param pipeline: the embedding pipeline containing the vectors in its table
    @type pipeline: EmbeddingPipeline
    @param resource_id_rev_list: a specific set of resource ids to get the embeddings from.
    If not specified, all embeddings in the embedding table of the pipeline will be returned
    @type resource_id_rev_list: Optional[List[Tuple[str, int]]])
    @return: a list of tuples (id, rev) for resources whose embedding vector could not be found,
    and a dict with key: (id, rev), value: embedding vector

    @rtype: Tuple[
            List[Tuple[str, int]],
            Dict[Tuple[str, int], List[float]]
        ]
    """

    embedding_table = pipeline.generate_embedding_table()

    embedding_table: Dict[Tuple, List] = dict(
        (parse_id_rev(key), value.tolist())
        for key, value in embedding_table.loc[:, "embedding"].to_dict().items()
    )

    if resource_id_rev_list is None:
        return [], embedding_table

    def get_from_embedding_table(resource_id: str, resource_rev: int) -> Tuple[Tuple, Optional[List[float]]]:
        if resource_rev is not None:
            return (
                (resource_id, resource_rev),
                embedding_table.get((resource_id, resource_rev), None)
            )
        else:
            return next(
                (
                    ((res_id, res_rev), embedding)
                    for (res_id, res_rev), embedding in embedding_table.items()
                    if res_id == resource_id
                ), ((resource_id, resource_rev), None)
            )

    computation: Dict[Tuple[str, int], Optional[List[float]]] = dict(
        get_from_embedding_table(resource_id, resource_rev)
        for resource_id, resource_rev in resource_id_rev_list
    )

    missing = [key for key, value in computation.items() if value is None]
    existing = dict((key, value) for key, value in computation.items() if value is not None)

    return missing, existing


def register_embeddings(
        forge: KnowledgeGraphForge, vectors: Dict[Tuple[str, int], List[float]],
        model_id: str, model_revision: int, embedding_tag: str,
        mapping_path: str,
) -> Tuple[str, int]:
    """
    Register and updates embedding vectors

    @param forge: a forge instance
    @type forge: KnowledgeGraphForge
    @param vectors: a dictionary with keys the entity ids + rev and the values the associated
    embedding vectors
    @type vectors: Dict[Tuple[str, int], List[int]]
    @param model_id: the id of the embedding model
    @type model_id: str
    @param model_revision: the revision of the embedding model
    @type model_revision: int
    @param embedding_tag:
    @type embedding_tag: str
    @param mapping_path: the path to a mapping indicating an embedding's format
    @type mapping_path: str
    # @param entity_type: the type of the entity that's been embedded
    # @type entity_type: str
    @return the tag applied to the embedding resources, and the vector dimension of the embeddings
    @rtype Tuple[str, int]
    """

    mapping = DictionaryMapping.load(mapping_path)

    new_embeddings: List[Resource] = []
    updated_embeddings: List[Resource] = []

    for (entity_id_i, entity_rev_i), embedding_vector_i in vectors.items():

        resource = forge.retrieve(entity_id_i)

        assert resource

        existing_vectors_i: List[Resource] = _search(
            entity_id=entity_id_i, forge=forge, model_id=model_id
        )

        # Embedding vector for this entity and this model exists, update it
        if existing_vectors_i is not None and len(existing_vectors_i) > 0:

            updated = _update(
                entity_id=entity_id_i, entity_rev=entity_rev_i,
                entity_type="NeuronMorphology",
                embedding=embedding_vector_i,
                model_id=model_id, model_revision=model_revision,
                embedding_vector_resource=existing_vectors_i[0]
            )
            updated_embeddings.append(updated)
        # Embedding vector for this entity and this model does not exist, create it
        else:

            created = _create(
                entity_id=entity_id_i, entity_rev=entity_rev_i, entity_type=resource.type,
                embedding=embedding_vector_i, forge=forge,
                model_id=model_id, model_revision=model_revision,
                mapping=mapping
            )
            new_embeddings.append(created)

    _persist(new_embeddings, True, forge=forge, tag=embedding_tag, obj_str="embeddings")
    _persist(updated_embeddings, False, forge=forge, tag=embedding_tag, obj_str="embeddings")

    vector_dimension = len(list(vectors.values())[0])

    return embedding_tag, vector_dimension


def _update(
        entity_id: str, entity_rev: int, entity_type: str, embedding: List[float],
        model_id: str, model_revision: int,
        embedding_vector_resource: Resource
) -> Resource:

    embedding_vector_resource.name = f"Embedding of {entity_id.split('/')[-1]} at revision {entity_rev}"
    embedding_vector_resource.embedding = embedding

    new_entity_dict = {
        "id": entity_id,
        "_rev": entity_rev,
        "type": entity_type
    }

    new_model_dict = {
        "id": model_id,
        "_rev": model_revision,
        "type": Types.EMBEDDING_MODEL.value
    }

    # def _idx_type(arr: List, type_: str, field_accessor: Callable[[Resource], str]) -> int:
    #     return next(
    #         i for i, res in enumerate(_enforce_list(arr)) if field_accessor(res) == type_
    #     )
    #
    # idx_generation_entity = _idx_type(
    #     embedding_vector_resource.generation.activity.used, entity_type, lambda res: res.type
    # )
    # idx_generation_model = _idx_type(
    #     embedding_vector_resource.generation.activity.used,
    #      Types.EMBEDDING_MODEL.value, lambda res: res.type
    # )

    # idx_derivation_entity = _idx_type(
    #     embedding_vector_resource.derivation, entity_type, lambda res: res.entity.type
    # )
    #
    # idx_derivation_model = _idx_type(
    #     embedding_vector_resource.derivation,
    #     Types.EMBEDDING_MODEL.value, lambda res: res.entity.type
    # )
    #
    #
    # embedding_vector_resource.derivation[idx_derivation_entity].entity = new_entity_dict
    # embedding_vector_resource.derivation[idx_derivation_model].entity = new_model_dict
    # embedding_vector_resource.generation.activity.used[idx_generation_model] = new_model_dict
    # embedding_vector_resource.generation.activity.used[idx_generation_entity] = new_entity_dict

    embedding_vector_resource.derivation = [
        {"entity": new_entity_dict}, {"entity": new_model_dict}
    ]
    embedding_vector_resource.generation.activity.used = [new_model_dict, new_entity_dict]

    embedding_vector_resource.generation.activity.wasAssociatedWith = [
        _software_agent_bluegraph(),
        _software_agent_similarity_tools()
    ]

    return embedding_vector_resource


def _create(
        entity_id: str,
        entity_rev: int,
        entity_type: str,
        embedding: List[float],
        forge: KnowledgeGraphForge,
        model_id: str,
        model_revision: int, mapping: Dict
) -> Resource:
    entity_dict = {
        "entity_id": entity_id,
        "entity_rev": entity_rev,
        "entity_type": entity_type,
        "model_id": model_id,
        "model_rev": model_revision,
        "embedding_name": f"Embedding of {entity_id.split('/')[-1]} at revision {entity_rev}",
        "embedding_vector": embedding,
        "embedding_id": create_id_with_forge(forge)
    }

    embedding: Resource = forge.map(entity_dict, mapping)

    embedding.generation.activity.wasAssociatedWith = [
        _software_agent_bluegraph(),
        _software_agent_similarity_tools()
    ]

    return embedding


def _search(
        entity_id: str, forge: KnowledgeGraphForge, model_id: str
) -> Optional[List[Resource]]:
    return forge.search({
        "type": Types.EMBEDDING.value,
        "derivation": {
            "entity": {
                "id": entity_id,
            }
        },
        "generation": {
            "activity": {
                "used": {
                    "id": model_id
                }
            }
        }
    })