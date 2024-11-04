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
