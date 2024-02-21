import numpy as np
import pandas as pd

from bluegraph import PandasPGFrame
from bluegraph.downstream import EmbeddingPipeline
from bluegraph.downstream.similarity import (FaissSimilarityIndex, ScikitLearnSimilarityIndex,
                                             SimilarityProcessor)

from similarity_tools.building.model_data_impl.neuron_morphologies import \
    NeuronMorphologies
from similarity_tools.data_classes.model import Model


class CoordinateModel(Model):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(dimension=3, similarity="euclidean")
        self.morphologies_df = model_data.morphologies_br_df
        self.full_df = model_data.full_df

    def run(self) -> EmbeddingPipeline:

        coordinate_df = pd.DataFrame(self.morphologies_df["id"])

        coordinate_df["coordinates"] = pd.Series(self.full_df[[
            "brainLocation.coordinatesInBrainAtlas.valueX.@value",
            "brainLocation.coordinatesInBrainAtlas.valueY.@value",
            "brainLocation.coordinatesInBrainAtlas.valueZ.@value"
        ]].values.tolist()).apply(lambda x: [float(el) for el in x])

        coordinate_df = coordinate_df.rename(columns={"id": "@id"})

        # Scale coordinates by dividing by the maximum value

        coordinates = np.array(coordinate_df["coordinates"].tolist())
        coordinate_df["coordinates"] = (coordinates / coordinates.max()).tolist()

        coordinate_df = coordinate_df.set_index("@id")

        # coordinates_frame = PandasPGFrame.from_frames(nodes=coordinate_df, edges=pd.DataFrame())

        similarity_index = FaissSimilarityIndex(
            dimension=self.dimension, similarity=self.similarity, n_segments=3
        )

        sim_processor = SimilarityProcessor(similarity_index, point_ids=None)

        point_ids = coordinate_df.index
        sim_processor.add(coordinate_df["coordinates"].tolist(), point_ids)

        pipeline = EmbeddingPipeline(
            preprocessor=None,
            embedder=None,
            similarity_processor=sim_processor
        )

        return pipeline
