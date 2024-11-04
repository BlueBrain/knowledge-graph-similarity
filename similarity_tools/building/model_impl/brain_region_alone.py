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

from abc import ABC
from typing import Dict, List, Tuple, Optional

import json
import os

from bluegraph import PandasPGFrame
from bluegraph.backends.gensim import GensimNodeEmbedder
from bluegraph.downstream import EmbeddingPipeline
from bluegraph.downstream.similarity import (ScikitLearnSimilarityIndex, SimilarityProcessor)

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.data_classes.model import Model
from similarity_tools.data_classes.model_data import ModelData
from similarity_tools.helpers.utils import encode_id_rev_resource


class BrModelData(ModelData):
    def __init__(
            self, bucket_configuration: NexusBucketConfiguration,
            src_data_dir=None, dst_data_dir=None
    ):
        super().__init__(
            org=bucket_configuration.organisation,
            project=bucket_configuration.project, deployment=bucket_configuration.deployment,
            src_data_dir=src_data_dir, dst_data_dir=dst_data_dir
        )

        self.bucket_configuration = bucket_configuration


class BBPBrainRegionModelAlone(Model, ABC):
    brain_region_hierarchy: Dict

    def __init__(self, model_data: BrModelData):
        super().__init__(dimension=32, similarity=None)
        self.src_data_dir = model_data.src_data_dir
        self.bucket_configuration = model_data.bucket_configuration
        self.brain_region_hierarchy = self.get_brain_region_hierarchy()

    def get_brain_region_hierarchy(self):
        with open(os.path.join(self.src_data_dir, "bmo/brainregion.json"), "r") as f:
            brain_hierarchy = json.load(f)["defines"]
            return dict((el["@id"], el) for el in brain_hierarchy)

    def run(self) -> EmbeddingPipeline:
        # Create a property graph from the loaded hierarchy

        def get_parent_child_pair(e) -> Optional[Tuple[str, str]]:

            if "hasHierarchyView" in e and \
                    'https://neuroshapes.org/BrainRegion' in e["hasHierarchyView"]:

                if "isPartOf" not in e:
                    return None

                parent_id = e["isPartOf"][0]
                parent = self.brain_region_hierarchy[parent_id]["@id"]

                return parent, e["@id"]

            return None

        edges = [
            e
            for e in list(map(get_parent_child_pair, self.brain_region_hierarchy.values()))
            if e is not None
        ]

        nodes = list(set(s for el in edges for s in el))

        forge = self.bucket_configuration.allocate_forge_session()

        br_res = list(map(forge.retrieve, nodes))

        br_id_rev = dict((e.id, encode_id_rev_resource(e)) for e in br_res)

        self.nodes = list(map(br_id_rev.get, nodes))

        self.edges = [(br_id_rev[a], br_id_rev[b]) for (a, b) in edges]

        frame = PandasPGFrame()
        frame.add_nodes(self.nodes)
        frame.add_edges(self.edges)

        # Train a Poincare embedding model for the hierarchy
        embedder = GensimNodeEmbedder(self.similarity, size=self.dimension, negative=2, epochs=100)
        embedding = embedder.fit_model(frame)

        # np.savetxt("brain_region_embs.tsv", np.array(embedding["embedding"].tolist()),
        #            delimiter="\t")

        similarity_index = ScikitLearnSimilarityIndex(
            dimension=self.dimension, similarity="euclidean",
            initial_vectors=embedding["embedding"].tolist()
        )

        point_ids = embedding.index
        sim_processor = SimilarityProcessor(similarity_index, point_ids=point_ids)

        pipeline = EmbeddingPipeline(
            preprocessor=None,
            embedder=None,
            similarity_processor=sim_processor
        )

        return pipeline
