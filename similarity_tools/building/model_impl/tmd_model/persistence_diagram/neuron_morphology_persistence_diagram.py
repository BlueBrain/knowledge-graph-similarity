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

from typing import Optional, List

from morphio import Morphology, Option
from tmd.Topology.methods import get_ph_neuron
from tmd.io.io import load_neuron_from_morphio

from kgforge.core import Resource, KnowledgeGraphForge

from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import \
    PersistenceDiagram, NeuriteType


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
    def get_persistence_data(cls, filename: str, neurite_type: NeuriteType) -> Optional[List]:

        try:
            morphology = Morphology(filename, Option.soma_sphere)
            neuron = load_neuron_from_morphio(morphology)

            return get_ph_neuron(
                neuron, neurite_type=neurite_type.value,
                feature=PersistenceDiagram.FILTRATION_METRIC
            )

        except Exception as e:
            print(f"{filename} failed")
            print(e)
            return None
