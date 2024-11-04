from typing import Dict, List, Tuple, Optional

import math
import json

import numpy as np
from kgforge.specializations.mappings import DictionaryMapping

from inference_tools.datatypes.similarity.statistic import Statistic
from inference_tools.similarity.formula import Formula
from similarity_tools.helpers.elastic import ElasticSearch

from kgforge.core import Resource, KnowledgeGraphForge

from similarity_tools.registration.helper_functions.common import _persist
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import SimilarityToolsException
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import create_id, create_id_with_forge
from similarity_tools.registration.helper_functions.software_agents import get_wasAssociatedWith


def _compute_score_deviation(forge, point_id, vector, score_min, score_max, k, formula: Formula):
    """
    Compute similarity score deviation for each vector
    @param forge:
    @type forge:
    @param point_id:
    @type point_id:
    @param vector:
    @type vector:
    @param score_min:
    @type score_min:
    @param score_max:
    @type score_max:
    @param k:
    @type k:
    @param formula:
    @type formula:
    @return:
    @rtype:
    """
    query = {
        "size": k,
        "query": {
            "script_score": {
                "query": {
                    "exists": {
                        "field": "embedding"
                    }
                },
                "script": {
                    "source": formula.get_formula(),
                    "params": {
                        "query_vector": vector
                    }
                }
            }
        }
    }

    def normalize(score, min_v, max_v):
        return (score - min_v) / (max_v - min_v)

    result = forge.elastic(json.dumps(query))
    if result is None:
        raise SimilarityToolsException("Score deviation")

    scores = set(
        normalize(el._store_metadata._score, score_min, score_max)
        for el in result if point_id != el.id
    )

    scores = 1 - np.array(list(scores))

    def spherical_gaussian_standard_deviation(value):
        return math.sqrt((value ** 2).mean())

    return spherical_gaussian_standard_deviation(scores)


def compute_boosting_factors(
        forge: KnowledgeGraphForge,
        stats: Statistic,
        formula: Formula,
        neighborhood_size: int = 10
) -> Dict[Tuple[str, int], float]:
    """
    Compute boosting factors for all vectors
    @param forge:
    @type forge: KnowledgeGraphForge
    @param stats:
    @type stats: Statistic
    @param formula:
    @type formula: Formula
    @param neighborhood_size:
    @type neighborhood_size: int
    @return:
    @rtype: Dict[Tuple[str, int], float]
    """

    def compute_boosting_factor(vector_resource: Resource) -> Tuple[Tuple[str, int], float]:
        key: Tuple[str, int] = vector_resource.id, vector_resource._store_metadata._rev

        # Compute local similarity deviations for points
        value: float = 1 + _compute_score_deviation(
            forge=forge, point_id=key,
            vector=vector_resource.embedding,
            score_min=stats.min, score_max=stats.max,
            k=neighborhood_size, formula=formula
        )

        return key, value

    all_vectors: List[Resource] = ElasticSearch.get_all_documents(forge)
    return dict(compute_boosting_factor(vector_resource) for vector_resource in all_vectors)


def register_boosting_factors(
        forge: KnowledgeGraphForge,
        view_id: str,
        boosting_factors: Dict[Tuple[str, int], float],
        formula: Formula,
        boosting_tag: str,
        mapping_path: str
):
    """
    Create similarity score boosting factor resources
    @param forge:
    @type forge:
    @param view_id: the boosting view id
    @type view_id: str
    @param boosting_factors: the boosting data to register, a dictionary with entity ids as keys,
    and boosting values as values
    @type boosting_factors: Dict[str, float]
    @param formula:
    @type formula: Formula
    @param mapping_path:
    @type mapping_path: str
    @param boosting_tag:
    @type boosting_tag: str
    @return:
    @rtype:
    """

    mapping = DictionaryMapping.load(mapping_path)
    boosting_factor_schema = forge._model.schema_id(Types.EMBEDDING.value)

    new_boosting_factors: List[Resource] = []
    updated_boosting_factors: List[Resource] = []

    for (e_id, e_rev), boosting_val in boosting_factors.items():
        # Look for boosting factor associated to this embedding id
        existing_data: Optional[Resource] = _search_per_embedding(embedding_id=e_id, forge=forge)

        # Boosting instance exists for this embedding id, update it
        if existing_data is not None:
            updated_boosting_resource = _update(
                existing_boosting_factor=existing_data,
                boosting_value=boosting_val,
                forge=forge,
                view_id=view_id,
                entity_rev=e_rev
            )
            updated_boosting_factors.append(updated_boosting_resource)
        else:
            # No boosting instance exists for this embedding id, create it
            created_boosting_resource = _create(
                entity_id=e_id,
                entity_rev=e_rev,
                boosting_value=boosting_val,
                mapping=mapping,
                formula=formula,
                view_id=view_id,
                forge=forge
            )

            new_boosting_factors.append(created_boosting_resource)

    _persist(new_boosting_factors, True, schema_id=boosting_factor_schema, forge=forge, tag=boosting_tag, obj_str="boosting factors")
    _persist(updated_boosting_factors, False, schema_id=boosting_factor_schema, forge=forge, tag=boosting_tag,
             obj_str="boosting factors")


def _update(
        existing_boosting_factor: Resource, boosting_value: float,
        forge: KnowledgeGraphForge, view_id: str, entity_rev: int
) -> Resource:
    generation_json = {
        "type": "Generation",
        "activity": {
            "type": "Activity",
            "used": {
                "id": view_id,
                "type": "ElasticSearchView"
            },
            "wasAssociatedWith": get_wasAssociatedWith(bluegraph=False)
        }
    }

    existing_boosting_factor.value = boosting_value
    existing_boosting_factor.generation = forge.from_json(generation_json)
    existing_boosting_factor.derivation.entity._rev = entity_rev

    return existing_boosting_factor


def _create(
        entity_id: str, boosting_value: float, mapping: Dict, formula: Formula,
        view_id: str, forge: KnowledgeGraphForge, entity_rev: int
) -> Resource:
    boosting_dict = {
        "boosting_value": boosting_value,
        "formula_str": formula.get_formula(),
        "entity_id": entity_id,
        "entity_rev": entity_rev,
        "view_id": view_id,
        "boosting_factor_id": create_id_with_forge(forge)
    }

    boosting_resource = forge.map(boosting_dict, mapping)

    boosting_resource.generation.activity.wasAssociatedWith = get_wasAssociatedWith(bluegraph=False)

    return boosting_resource


def _search_per_embedding(embedding_id: str, forge: KnowledgeGraphForge) -> \
        Optional[Resource]:
    res = forge.search({
        "type": Types.SIMILARITY_BOOSTING_FACTOR.value,
        "derivation": {
            "entity": {
                "id": embedding_id
            }
        }
    })

    if res is None:
        raise SimilarityToolsException("Error fetching existing similarity boosting factor")

    if len(res) == 0:
        return None

    if len(res) > 1:
        logger.warning(
            "Warning Multiple boosting factors found for one embedding under the same tag, "
            "this should not happen, only getting first one"
        )

    return res[0]
