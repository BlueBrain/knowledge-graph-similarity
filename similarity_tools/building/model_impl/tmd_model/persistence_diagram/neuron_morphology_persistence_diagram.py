from typing import Optional, List

from morphio import Morphology, Option
from tmd.Topology.methods import get_ph_neuron
from tmd.io.io import load_neuron_from_morphio

from kgforge.core import Resource, KnowledgeGraphForge

from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import \
    PersistenceDiagram


class NeuronMorphologyPersistenceDiagram(PersistenceDiagram):
    @classmethod
    def get_distribution(cls, m: Resource, forge: KnowledgeGraphForge) -> Optional[Resource]:

        def as_list(el):
            return el if isinstance(el, list) else [el]
        try:
            return next(d for d in as_list(m.distribution) if d.name.endswith(".swc"))
        except StopIteration:
            return None

    @classmethod
    def get_persistence_data(cls, filename: str) -> Optional[List]:

        try:
            morphology = Morphology(filename, Option.soma_sphere)
            neuron = load_neuron_from_morphio(morphology)

            return get_ph_neuron(
                neuron, neurite_type=PersistenceDiagram.NEURITE_TYPE,
                feature=PersistenceDiagram.FILTRATION_METRIC
            )

        except Exception as e:
            print(f"{filename} failed")
            print(e)
            return None
