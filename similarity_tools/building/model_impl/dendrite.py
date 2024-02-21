from bluegraph import PandasPGFrame
from bluegraph.preprocess import ScikitLearnPGEncoder
from bluegraph.preprocess import CooccurrenceGenerator
from bluegraph.downstream import EmbeddingPipeline
# from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
# from bluegraph.downstream.similarity import (FaissSimilarityIndex, SimilarityProcessor)
from similarity_tools.building.model_data_impl.neuron_morphologies import \
    NeuronMorphologies, NeuronMorphologiesLoad

from similarity_tools.data_classes.model import Model
import pandas as pd


class DendriteModel(Model):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(dimension=100, similarity="cosine")
        self.location_features = model_data.location_features

        dendrite_frame = PandasPGFrame.from_frames(
            nodes=self.location_features, edges=pd.DataFrame())

        compartments = set(model_data.compartments_with_location_features) - \
                       set(model_data.compartments_to_exclude)

        properties_props = set(
            model_data.compartment_feature_name_to_str(c, f)
            for c in compartments
            for f in model_data.location_feature_names
        )

        props = set(dendrite_frame.node_properties()).difference(properties_props)

        encoder = ScikitLearnPGEncoder(
            node_properties=props,
            missing_numeric="impute",
            imputation_strategy="mean",
            reduce_node_dims=True,
            n_node_components=40
        )

        encoded_frame = encoder.fit_transform(dendrite_frame)

        # print(sum(encoder.node_reducer.explained_variance_ratio_))

        # Generate the dendrite co-projection graph

        gen = CooccurrenceGenerator(dendrite_frame)

        dendrite_edges = gen.generate_from_nodes(
            "BasalDendrite_Leaf_Regions", compute_statistics=["frequency"]
        )

        # print(dendrite_edges.shape)

        dendrite_co_projection_frame = PandasPGFrame.from_frames(
            nodes=encoded_frame._nodes, edges=dendrite_edges
        )

        dendrite_co_projection_frame.edge_prop_as_numeric("frequency")

        self.dendrite_co_projection_frame = dendrite_co_projection_frame

        # print(len(dendrite_co_projection_frame.isolated_nodes()))

    def run(self) -> EmbeddingPipeline:

        return None
        # Perform dendrite co-occurrence graph embedding
        # dendrite_embedder = StellarGraphNodeEmbedder(
        #     "node2vec", length=5, number_of_walks=20,
        #     epochs=5, embedding_dimension=self.dimension, edge_weight="frequency",
        #     random_walk_p=2, random_walk_q=0.2
        # )
        #
        # dendrite_embedding = dendrite_embedder.fit_model(self.dendrite_co_projection_frame)
        #
        # self.dendrite_co_projection_frame.add_node_properties(
        #     dendrite_embedding.rename(columns={"embedding": "node2vec"})
        # )
        #
        # sim_processor = SimilarityProcessor(
        #     similarity_index=FaissSimilarityIndex(
        #         dimension=self.dimension, similarity=self.similarity, n_segments=3,
        #         initial_vectors=dendrite_embedding["embedding"]
        #     ),
        #     point_ids=dendrite_embedding.index
        # )
        #
        # return EmbeddingPipeline(
        #     embedder=dendrite_embedder,
        #     similarity_processor=sim_processor
        # )


if __name__ == "__main__":
    a = NeuronMorphologiesLoad(
        org="bbp-external", project="seu", save_dir="../model_data_impl/test"
    )
    e = DendriteModel(a).run()
