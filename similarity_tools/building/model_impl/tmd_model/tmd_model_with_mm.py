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

import numpy as np
import os

from bluegraph.downstream import EmbeddingPipeline

from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import NeuriteType
# from bluegraph.downstream.similarity import (ScikitLearnSimilarityIndex, SimilarityProcessor)

from similarity_tools.data_classes.model import Model
from similarity_tools.building.model_data_impl.morphology_models import ModelDataMorphologyModels
from similarity_tools.building.model_impl.tmd_model.persistence_diagram\
    .morphology_model_persistence_diagram import MorphologyModelPersistenceDiagram
from similarity_tools.building.model_impl.tmd_model.vectorisation import Vectorisation


class TMDModelWithMM(Model):

    def __init__(self, model_data: ModelDataMorphologyModels, re_compute: bool, re_download: bool):

        env = model_data.deployment.name.lower()

        persistence_diagram_dir = os.path.join(
            model_data.src_data_dir, "persistence_diagrams_morphology_models"
        )
        persistence_diagram_location = os.path.join(
            persistence_diagram_dir, f"persistence_diagrams_{env}.json"
        )

        download_dir = os.path.join(model_data.src_data_dir, "morphology_models", env)

        ist = MorphologyModelPersistenceDiagram()

        self.model_persistence_diagrams = ist.get_persistence_diagrams(
                re_download=re_download,
                forge=model_data.forge,
                data=model_data.data,
                download_dir=download_dir,
                persistence_diagram_location=persistence_diagram_location,
                re_compute=re_compute,
                neurite_type=NeuriteType.BASAL_DENDRITE
            )

    def run(self) -> EmbeddingPipeline:

        return TMDModelWithMM.rest(
            self.model_persistence_diagrams,
            dim=self.dim,
            max_time=self.max_time,
            kernel_width=self.kernel_width,
            max_height=self.max_height
        )

    @staticmethod
    def rest(nm_persistence_diagrams, dim, max_time, kernel_width, max_height) -> EmbeddingPipeline:

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

        # keys = list(vectors.keys())
        x = np.float32(np.stack(list(vectors.values())))  # TODO check
        x = (x / x.max()).tolist()

        # TODO uncomment
        return x

        # similarity_index = ScikitLearnSimilarityIndex(
        #     dimension=dim, similarity="euclidean",
        #     initial_vectors=x
        # )
        #
        # pipeline = EmbeddingPipeline(
        #     preprocessor=None,
        #     embedder=None,
        #     similarity_processor=SimilarityProcessor(similarity_index, point_ids=keys)
        # )

        # return pipeline

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


class UnscaledTMDModelWithMM(TMDModelWithMM):

    def __init__(self, model_data: ModelDataMorphologyModels, re_compute=False, re_download=False):
        super().__init__(model_data, re_compute, re_download)

        self.dim = 256
        self.kernel_width = 120
        self.max_height = 17000

        # Compute maximum death/birth time of all diagrams to know the global scale.
        all_maxes = []
        for d in self.model_persistence_diagrams.values():
            d = np.array(d)
            all_maxes += [d[:, 0].max(), d[:, 1].max()]

        self.max_time = max(all_maxes)


class ScaledTMDModel(TMDModelWithMM):

    def __init__(self, model_data: ModelDataMorphologyModels, re_compute=False, re_download=False):
        super().__init__(model_data, re_compute, re_download)

        self.dim = 256
        self.max_height = 7
        self.max_time = 1
        self.kernel_width = 0.02

        # 5.2. Scale persistence diagrams before vectorization
        # Scale each diagram, so that the birth/death time are in the interval [0, 1].

        scaled_nm_persistence_diagrams = {}
        for name, diagram in self.model_persistence_diagrams.items():
            diagram = np.array(diagram)
            t_min = diagram.min()
            t_max = diagram.max()
            scaled_nm_persistence_diagrams[name] = (diagram - t_min) / (t_max - t_min)

        self.nm_persistence_diagrams = scaled_nm_persistence_diagrams
