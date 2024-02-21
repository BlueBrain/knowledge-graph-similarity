from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import json
from os.path import join

from bluegraph import PandasPGFrame
from bluegraph.backends.gensim import GensimNodeEmbedder
from bluegraph.downstream import EmbeddingPipeline
from bluegraph.downstream.similarity import (ScikitLearnSimilarityIndex, SimilarityProcessor)
from pandas import DataFrame

from similarity_tools.building.model_data_impl.neuron_morphologies import \
    NeuronMorphologies
from similarity_tools.data_classes.model import Model


class BrainRegionModel(Model, ABC):
    brain_region_hierarchy: Dict

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(similarity="poincare")
        self.src_data_dir = model_data.src_data_dir
        self.org = model_data.org
        self.project = model_data.project
        self.morphologies_br_df = model_data.morphologies_br_df

    @abstractmethod
    def get_brain_region_hierarchy(self):
        pass

    def run(self) -> Optional[EmbeddingPipeline]:

        brain_region_hierarchy_frame = PandasPGFrame()
        brain_region_hierarchy_frame.add_nodes(self.nodes)
        brain_region_hierarchy_frame.add_edges(self.edges)

        # Train a Poincare embedding model for the hierarchy
        embedder = GensimNodeEmbedder(self.similarity, size=32, negative=2, epochs=100)
        brain_region_hierarchy_embeddings = embedder.fit_model(brain_region_hierarchy_frame)

        # np.savetxt("brain_region_embs.tsv", np.array(embedding["embedding"].tolist()),
        #            delimiter="\t")

        brain_region_dimension = brain_region_hierarchy_embeddings["embedding"].iloc[0].shape[0]

        # df = embedding.reset_index()[["@id"]]
        # df["label"] = df["@id"]
        # df.to_csv("brain_region_meta.tsv", sep="\t", index=None)

        nm_embeddings = DataFrame(self.frame[
            self.frame["br"].isin(brain_region_hierarchy_embeddings.index.values)
        ])

        could_not = set(self.frame.index) - set(nm_embeddings.index)

        print("Total number of neuron morphologies:", len(self.frame))
        print("Neuron morphologies with brain region in the bbp hierarchy:", len(nm_embeddings))
        print("Neuron morphologies that could not be embedded", could_not)
        print("Brain regions not found", [self.frame.loc[nm_id, "br"] for nm_id in could_not])

        nm_embeddings.loc[:, "embedding"] = nm_embeddings.loc[:, "br"].apply(
            lambda x: brain_region_hierarchy_embeddings.loc[x, "embedding"]
        )

        if len(nm_embeddings) == 0:
            print(f"No embedding vectors were computed for {self.org}/{self.project}")
            return None

        similarity_index = ScikitLearnSimilarityIndex(
            dimension=brain_region_dimension, similarity="euclidean",
            initial_vectors=nm_embeddings["embedding"].tolist()
        )

        point_ids = nm_embeddings.index
        sim_processor = SimilarityProcessor(similarity_index, point_ids=point_ids)

        pipeline = EmbeddingPipeline(
            preprocessor=None,
            embedder=None,
            similarity_processor=sim_processor
        )

        return pipeline


class AllenBrainRegionModel(BrainRegionModel):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(model_data)

        def _get_children(hierarchy, edges, father=None):
            for child in hierarchy['children']:
                acronym = child["acronym"]
                if father:
                    edges.append((acronym, father))
                _get_children(child, edges, acronym)

        self.edges = []
        _get_children(self.get_brain_region_hierarchy(), self.edges)

        self.nodes = list(set([s for el in self.edges for s in el]))

    def get_brain_region_hierarchy(self):
        with open(join(self.src_data_dir, "1.json"), "r") as f:
            allen_hierarchy = json.load(f)
            return allen_hierarchy["msg"][0]

        # Create a property graph from the loaded hierarchy


class BBPBrainRegionModel(BrainRegionModel):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(model_data)

        def _get_children(hierarchy, edges, father=None):
            for child in hierarchy['children']:
                acronym = child["acronym"]
                if father:
                    edges.append((acronym, father))
                _get_children(child, edges, acronym)

        self.edges = []
        _get_children(self.get_brain_region_hierarchy(), self.edges)

        self.nodes = list(set([s for el in self.edges for s in el]))

    def get_brain_region_hierarchy(self):
        with open(join(self.src_data_dir, "atlas/mba_hierarchy_v3l23split.json"), "r") as f:
            allen_hierarchy = json.load(f)
            return allen_hierarchy["msg"][0]


class BBPBrainRegionBMOModel(BrainRegionModel):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(model_data)

        brain_hierarchy = self.get_brain_region_hierarchy()

        def get_parent_child_pair(e) -> Optional[Tuple[str, str]]:

            if "hasHierarchyView" in e and \
                    'https://neuroshapes.org/BrainRegion' in e["hasHierarchyView"]:

                if "isPartOf" not in e:
                    return None

                parent_id = e["isPartOf"][0]
                parent = brain_hierarchy[parent_id]["notation"]

                return parent, e["notation"]

            return None

        edges = [get_parent_child_pair(e) for e in brain_hierarchy.values()]
        self.edges = [e for e in edges if e is not None]

        self.nodes = list(set([s for el in self.edges for s in el]))

    def get_brain_region_hierarchy(self):
        with open(join(self.src_data_dir, "bmo/brainregion.json"), "r") as f:
            brain_hierarchy = json.load(f)["defines"]
            return dict((el["@id"], el) for el in brain_hierarchy)
