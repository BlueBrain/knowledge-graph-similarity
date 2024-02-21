from typing import List, Dict, Tuple, Optional, Union
from kgforge.core import Resource, KnowledgeGraphForge
import numpy as np

from inference_tools.datatypes.similarity.neighbor import Neighbor
from inference_tools.datatypes.similarity.statistic import Statistic
from inference_tools.similarity.formula import Formula
from inference_tools.similarity.queries.get_neighbors import get_neighbors
from similarity_tools.helpers.elastic import ElasticSearch
from similarity_tools.registration.helper_functions.software_agents import \
    _software_agent_similarity_tools, _software_agent_bluegraph
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import SimilarityToolsException
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import create_id, create_id_with_forge, raise_error_on_failure


def compute_statistics(
        forge: KnowledgeGraphForge,
        score_formula: Formula,
        derivation_type: str,
        boosting: Dict[str, float] = None,
) -> Statistic:
    """Compute similarity score statistics given a view."""

    all_vectors: List[Resource] = ElasticSearch.get_all_documents(forge)

    scores = []

    for i, vector_resource in enumerate(all_vectors):
        if i % 20 == 0:
            logger.info(f">  Neighbors computed for {i}/{len(all_vectors)} neuron morphologies")

        neighbors: List[Tuple[int, Optional[Neighbor]]] = get_neighbors(
            vector=vector_resource.embedding,
            forge=forge,
            vector_id=vector_resource.id,
            k=len(all_vectors),
            score_formula=score_formula,
            use_forge=False,
            debug=False,
            derivation_type=derivation_type
        )

        boosting_value = boosting[vector_resource.id] if boosting else 1

        scores += [score * boosting_value for score, _ in neighbors]

    scores = np.array(scores)
    return Statistic(scores.min(), scores.max(), scores.mean(), scores.std(), float(len(scores)))


def register_stats(
        forge: KnowledgeGraphForge,
        aggregated_similarity_view_id: str,
        stats: Statistic,
        mapping: Dict,
        formula: Formula,
        stats_tag: str,
        boosted: bool = False
) -> str:
    """Create ES view statistic resources."""

    series = stats.to_series()

    # Check if a statistics view for this entity id already exists
    stats_resource: Optional[Resource] = _search_stats(
        forge=forge, boosted=boosted, aggregated_similarity_view_id=aggregated_similarity_view_id,
        as_resource=True
    )

    if stats_resource is not None:
        stats_resource.series = series
        stats_resource.derivation.entity.id = aggregated_similarity_view_id
        stats_resource.scriptScore = formula.value
        forge.update(stats_resource)
    else:
        json_data = {
            "stat_id": create_id_with_forge(forge),
            "boosted": boosted,
            "formula_value": formula.value,
            "series": series,
            "view_id": aggregated_similarity_view_id
        }

        stats_resource = forge.map(json_data, mapping)
        stats_resource.generation.activity.wasAssociatedWith = _software_agent_similarity_tools()

        forge.register(stats_resource)

    raise_error_on_failure(stats_resource)
    forge.tag(stats_resource, stats_tag)
    raise_error_on_failure(stats_resource)

    return stats_resource.get_identifier()


def _search_stats(
        forge: KnowledgeGraphForge, boosted: bool, aggregated_similarity_view_id: str,
        as_resource: bool = False
) -> Optional[Union[Statistic, Resource]]:

    res = forge.search({
        "type": Types.ES_VIEW_STATS.value,
        "boosted": boosted,
        "derivation": {
            "entity": {
                "id": aggregated_similarity_view_id
            }
        }
    })

    if res is None:
        raise SimilarityToolsException("Could not fetch statistics")

    if len(res) == 0:
        return None

    if len(res) > 1:
        logger.warning("Warning Multiple statistics found, this should not happen, only getting "
                       "the first one")

    return Statistic.from_json(forge.as_json(res[0])) if not as_resource else res[0]


