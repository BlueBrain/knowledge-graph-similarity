from bluegraph import PandasPGFrame
from bluegraph.preprocess import ScikitLearnPGEncoder
from bluegraph.preprocess import CooccurrenceGenerator

from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.downstream.similarity import (FaissSimilarityIndex, SimilarityProcessor)
from bluegraph.downstream import EmbeddingPipeline

from similarity_tools.building.model_data_impl.neuron_morphologies import \
    NeuronMorphologies, NeuronMorphologiesLoad
from similarity_tools.data_classes.model import Model
import pandas as pd


class AxonModel(Model):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(similarity="cosine", dimension=128)

        self.location_features = model_data.location_features

        axon_frame = PandasPGFrame.from_frames(nodes=self.location_features, edges=pd.DataFrame())

        compartments = set(model_data.compartments_with_location_features) - \
                       set(model_data.compartments_to_exclude)

        properties_props = set(
            model_data.compartment_feature_name_to_str(c, f)
            for c in compartments
            for f in model_data.location_feature_names
        )

        props = set(axon_frame.node_properties()).difference(properties_props)

        encoder = ScikitLearnPGEncoder(
            node_properties=props,
            missing_numeric="impute",
            imputation_strategy="mean",
            reduce_node_dims=True,
            n_node_components=40
        )
        # essentially encodes ApicalDendrite_Section/Leaf_Regions and brain region id properties
        encoded_frame = encoder.fit_transform(axon_frame)

        # print(sum(encoder.node_reducer.explained_variance_ratio_))

        # Generate the axon co-projection graph

        gen = CooccurrenceGenerator(axon_frame)

        # Edges -> pairs of neuron morphology ids
        # Per edge, two properties, common - List of BR they have in common, frequency=number of
        # times they're both repeated in the region
        axon_edges = gen.generate_from_nodes(
            "Axon_Leaf_Regions", compute_statistics=["frequency"]
        )

        axon_edges = axon_edges[axon_edges["frequency"].values > 10]

        axon_co_projection_frame = PandasPGFrame.from_frames(
            nodes=encoded_frame._nodes, edges=axon_edges
        )

        axon_co_projection_frame.edge_prop_as_numeric("frequency")
        self.axon_co_projection_frame = axon_co_projection_frame

    def run(self) -> EmbeddingPipeline:
        axon_embedder = StellarGraphNodeEmbedder(
            "node2vec", length=5, number_of_walks=20,
            epochs=5, embedding_dimension=self.dimension,
            edge_weight="frequency",
            random_walk_p=2, random_walk_q=0.2
        )

        axon_embedding = axon_embedder.fit_model(self.axon_co_projection_frame)

        self.axon_co_projection_frame.add_node_properties(
            axon_embedding.rename(columns={"embedding": "node2vec"})
        )

        sim_processor = SimilarityProcessor(
            similarity_index=FaissSimilarityIndex(
                dimension=self.dimension, similarity=self.similarity, n_segments=3,
                initial_vectors=axon_embedding["embedding"]
            ),
            point_ids=axon_embedding.index
        )

        pipeline = EmbeddingPipeline(
            embedder=axon_embedder,
            similarity_processor=sim_processor
        )

        return pipeline


if __name__ == "__main__":
    a = NeuronMorphologiesLoad(
        org="bbp-external", project="seu", save_dir="../model_data_impl/test"
    )
    e = AxonModel(a).run()
