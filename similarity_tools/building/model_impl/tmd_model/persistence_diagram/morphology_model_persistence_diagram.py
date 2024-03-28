from kgforge.core import KnowledgeGraphForge, Resource
from typing import Optional, List

from similarity_tools.registration.registration_exception import SimilarityToolsException

from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import \
    PersistenceDiagram, NeuriteType

import json


class MorphologyModelPersistenceDiagram(PersistenceDiagram):

    @classmethod
    def get_distribution(cls, m: Resource, forge: KnowledgeGraphForge) -> Optional[Resource]:
        return forge.retrieve(m.morphologyModelDistribution.id).distribution

    @classmethod
    def get_persistence_data(cls, filename, neurite_type: NeuriteType) -> Optional[List]:
        with open(filename) as f:
            t = json.load(f)

        compartment = t[neurite_type.value]

        if compartment["filtration_metric"] != PersistenceDiagram.FILTRATION_METRIC:

            raise SimilarityToolsException(
                f"The persistence diagram in {PersistenceDiagram.NEURITE_TYPE} does not have "
                f"the required filtration metric: expected {PersistenceDiagram.FILTRATION_METRIC}, "
                f"got {compartment['filtration_metric']} "
            )

        def get_filtration_value_only(tree):
            # https://github.com/BlueBrain/TMD/blob/1fe329eaea05b8ec016ab226859d1a752485cdc9/tmd/Topology/methods.py#L34
            # filtration value start and end only
            return [arr_i[:2] for arr_i in tree]

        return list(map(get_filtration_value_only, compartment["persistence_diagram"]))
