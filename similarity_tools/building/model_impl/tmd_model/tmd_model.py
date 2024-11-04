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

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Union
from typing_extensions import Unpack

import numpy as np
import os

from bluegraph.downstream import EmbeddingPipeline

from bluegraph.downstream.similarity import (ScikitLearnSimilarityIndex, SimilarityProcessor)

from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import \
    NeuriteType
from similarity_tools.data_classes.model import Model
from similarity_tools.building.model_data_impl.neuron_morphologies_query import \
    NeuronMorphologiesQuery

from similarity_tools.building.model_impl.tmd_model.persistence_diagram \
    .neuron_morphology_persistence_diagram import NeuronMorphologyPersistenceDiagram

from similarity_tools.building.model_impl.tmd_model.vectorisation import Vectorisation
from enum import Enum
from tmd.Topology.vectorizations import get_limits


class VectorisationTechnique(Enum):
    PERSISTENCE_IMAGE_DATA = 1
    BETTI_CURVE = 2
    LIFE_ENTROPY_CURVE = 3


class TMDModel(Model, ABC):

    def __init__(
            self,
            model_data: NeuronMorphologiesQuery,
            re_compute: bool,
            re_download: bool,
            neurite_type: NeuriteType
    ):
        persistence_diagram_directory = os.path.join(
            model_data.src_data_dir, "persistence_diagrams"
        )

        append_env = model_data.deployment.name.lower()

        persistence_diagram_location = os.path.join(
            persistence_diagram_directory,
            f"persistence_diagrams_{neurite_type.value}_{model_data.org}_{model_data.project}_"
            f"{append_env}.json"
        )

        download_dir = os.path.join(
            model_data.src_data_dir,
            f"morphologies_{model_data.org}_{model_data.project}_{append_env}",
        )

        self.nm_persistence_diagrams: Dict[str, List] = NeuronMorphologyPersistenceDiagram.get_persistence_diagrams(
            re_download=re_download,
            re_compute=re_compute,
            download_dir=download_dir,
            persistence_diagram_location=persistence_diagram_location,
            forge=model_data.forge,
            data=model_data.data,
            neurite_type=neurite_type
        )

        self.nm_persistence_diagrams: Dict[str, List] = dict(
            (key, value)
            for key, value in self.nm_persistence_diagrams.items()
            if set(len(val_i) for val_i in value) == {2}  # TODO ??? What is this??
        )

    @abstractmethod
    def run(self) -> EmbeddingPipeline:
        pass


class TMDModelNew(TMDModel):

    vectorisation_technique: VectorisationTechnique

    def __init__(
            self,
            model_data: NeuronMorphologiesQuery,
            re_compute: bool,
            re_download: bool,
            neurite_type: NeuriteType,
    ):
        super().__init__(model_data, re_compute, re_download, neurite_type)

    def run(self) -> Union[EmbeddingPipeline, Dict[str, List]]:

        xlim, ylim = get_limits(list(self.nm_persistence_diagrams.values()))

        return TMDModelNew.run_static(
            vectorisation_technique=self.vectorisation_technique,
            nm_persistence_diagrams=self.nm_persistence_diagrams,
            xlim=xlim,
            ylim=ylim
        )

    @staticmethod
    def run_static(
            vectorisation_technique: VectorisationTechnique,
            nm_persistence_diagrams: Dict,
            xlim,
            ylim,
            **kwargs
    ) -> Union[EmbeddingPipeline, Dict[str, List]]:

        tech_to_method: Dict[VectorisationTechnique, Callable[[Unpack], Callable]] = {
            VectorisationTechnique.PERSISTENCE_IMAGE_DATA: Vectorisation.persistence_image_data,
            VectorisationTechnique.BETTI_CURVE: Vectorisation.betti_curve,
            VectorisationTechnique.LIFE_ENTROPY_CURVE: Vectorisation.life_entropy_curve
        }

        method = tech_to_method[vectorisation_technique](xlim=xlim, ylim=ylim)

        vectors = dict()

        for morphology_id, diagram in nm_persistence_diagrams.items():
            try:
                vectors[morphology_id] = method(diagram)
            except Exception as e:
                print(f"Failed vectorisation for {morphology_id}: {e}")

        return vectors


class TMDModelOld(TMDModel):
    dim: int
    max_time: float
    kernel_width: float
    max_height: int

    @staticmethod
    def old_vectorisation_run(
            nm_persistence_diagrams, dim, max_time, kernel_width, max_height
    ) -> EmbeddingPipeline:
        vectors = dict(
            (
                morphology_id,
                Vectorisation.compute_persistence_vector(
                    diagram=diagram,
                    dim=dim,
                    max_time=max_time,
                    kernel_width=kernel_width,
                    max_height=max_height
                )
            )
            for morphology_id, diagram in nm_persistence_diagrams.items()
        )

        keys = list(vectors.keys())
        x = np.float32(np.stack(list(vectors.values())))  # TODO check
        x = (x / x.max()).tolist()

        similarity_index = ScikitLearnSimilarityIndex(
            dimension=dim, similarity="euclidean",
            initial_vectors=x
        )

        pipeline = EmbeddingPipeline(
            preprocessor=None,
            embedder=None,
            similarity_processor=SimilarityProcessor(similarity_index, point_ids=keys)
        )

        return pipeline

        # TODO uncomment

        # Actually, for this kind of embeddings, it makes more sense to use
        # [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric).
        # See BlueGraph implementation:

        # similarity_index = ScikitLearnSimilarityIndex(
        #     dimension=dim, similarity="wasserstein",
        #     initial_vectors=scaled_x)
        # sim_processor = SimilarityProcessor(
        #     similarity_index, keys)
        # pipeline = EmbeddingPipeline(
        #     preprocessor=None,
        #     embedder=None,
        #     similarity_processor=sim_processor)


class UnscaledTMDModel(TMDModelOld):

    def run(self) -> EmbeddingPipeline:
        return UnscaledTMDModel.old_vectorisation_run(
            self.nm_persistence_diagrams,
            dim=self.dim,
            max_time=self.max_time,
            kernel_width=self.kernel_width,
            max_height=self.max_height
        )

    def __init__(self, model_data: NeuronMorphologiesQuery, re_compute=True, re_download=True):
        super().__init__(model_data=model_data, re_compute=re_compute, re_download=re_download)

        self.dim = 256
        self.kernel_width = 120
        self.max_height = 17000

        # Equivalent to code below, right?
        flattened = [
            b
            for persistent_diagram in self.nm_persistence_diagrams.values()
            for a in persistent_diagram
            for b in a
        ]

        self.max_time = max(flattened)

        # # Compute maximum death/birth time of all diagrams to know the global scale.
        # all_maxes = []
        # for d in self.nm_persistence_diagrams.values():
        #     d = np.array(d)
        #     all_maxes += [d[:, 0].max(), d[:, 1].max()]
        #
        # self.max_time = max(all_maxes)


class ScaledTMDModel(TMDModelOld):

    def run(self) -> EmbeddingPipeline:
        return ScaledTMDModel.old_vectorisation_run(
            self.nm_persistence_diagrams,
            dim=self.dim,
            max_time=self.max_time,
            kernel_width=self.kernel_width,
            max_height=self.max_height
        )

    def __init__(self, model_data: NeuronMorphologiesQuery, re_compute=True, re_download=True):
        super().__init__(model_data=model_data, re_compute=re_compute, re_download=re_download)

        self.dim = 256
        self.max_height = 7
        self.max_time = 1
        self.kernel_width = 0.02

        # 5.2. Scale persistence diagrams before vectorization
        # Scale each diagram, so that the birth/death time are in the interval [0, 1].

        def scale(diagram):
            diagram = np.array(diagram)
            t_min = diagram.min()
            t_max = diagram.max()
            return (diagram - t_min) / (t_max - t_min)

        self.nm_persistence_diagrams = dict(
            (name, scale(diagram))
            for name, diagram in self.nm_persistence_diagrams.items()
        )
