import numpy as np
import os

from bluegraph.downstream import EmbeddingPipeline

from bluegraph.downstream.similarity import (ScikitLearnSimilarityIndex, SimilarityProcessor)

from similarity_tools.data_classes.model import Model
from similarity_tools.building.model_data_impl.neuron_morphologies_query import \
    NeuronMorphologiesQuery

from similarity_tools.building.model_impl.tmd_model.persistence_diagram\
    .neuron_morphology_persistence_diagram import NeuronMorphologyPersistenceDiagram

from similarity_tools.building.model_impl.tmd_model.vectorisation import Vectorisation


class TMDModel(Model):
    dim: int
    max_time: float
    kernel_width: float
    max_height: int

    def __init__(self, model_data: NeuronMorphologiesQuery, re_compute: bool, re_download: bool):

        persistence_diagram_directory = os.path.join(
            model_data.src_data_dir, "persistence_diagrams"
        )

        append_env = model_data.deployment.name.lower()

        persistence_diagram_location = os.path.join(
            persistence_diagram_directory,
            f"persistence_diagrams_all_neurites_{model_data.org}_{model_data.project}_"
            f"{append_env}.json"
        )

        download_dir = os.path.join(
            model_data.src_data_dir,
            f"morphologies_{model_data.org}_{model_data.project}_{append_env}",
        )

        self.nm_persistence_diagrams = NeuronMorphologyPersistenceDiagram.get_persistence_diagrams(
                re_download=re_download,
                re_compute=re_compute,
                download_dir=download_dir,
                persistence_diagram_location=persistence_diagram_location,
                forge=model_data.forge,
                data=model_data.data
            )

        self.nm_persistence_diagrams = dict(
            (key, value)
            for key, value in self.nm_persistence_diagrams.items()
            if set(len(val_i) for val_i in value) == {2}
        )

    def run(self) -> EmbeddingPipeline:

        return TMDModel.rest(
            self.nm_persistence_diagrams,
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
                Vectorisation.build_vectors_from_tmd_implementations(diagram)
                # Vectorisation.compute_persistence_vector(
                #     diagram=diagram,
                #     dim=dim,
                #     max_time=max_time,
                #     kernel_width=kernel_width,
                #     max_height=max_height
                # )
            )
            for morphology_id, diagram in nm_persistence_diagrams.items()
        )

        return vectors
        # keys = list(vectors.keys())
        # x = np.float32(np.stack(list(vectors.values())))  # TODO check
        # x = (x / x.max()).tolist()
        #
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
        #
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


class UnscaledTMDModel(TMDModel):

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


class ScaledTMDModel(TMDModel):

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
