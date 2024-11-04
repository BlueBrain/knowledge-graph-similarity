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
from bluegraph.preprocess import ScikitLearnPGEncoder
from bluegraph.downstream import EmbeddingPipeline
from bluegraph.downstream.similarity import (FaissSimilarityIndex, SimilarityProcessor)

from similarity_tools.building.model_data_impl.neuron_morphologies_load import NeuronMorphologiesLoad
from similarity_tools.data_classes.model import Model
from similarity_tools.building.model_data_impl.neuron_morphologies import \
    NeuronMorphologies
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment
from similarity_tools.helpers.logger import logger


class NeuriteModel(Model):

    def __init__(self, model_data: NeuronMorphologies):
        super().__init__(similarity="euclidean", dimension=None)

        logger.info("Extract neurite features")

        self.compartments_to_exclude = model_data.compartments_to_exclude
        self.statistics_of_interest = ["mean", "standard deviation"]

        filter_out_0_annotations = [
            len(model_data.annotations[id_][1]) != 0
            for id_ in model_data.full_df['id']
        ]

        neurite_features_base = model_data.full_df.loc[filter_out_0_annotations]

        tmp = neurite_features_base.apply(
            lambda x: self.get_neurom_feature_annotations(x, model_data), axis=1
        ).set_index("id")

        self.neurite_features_df = pd.concat([model_data.morphologies_br_df, tmp], axis=1)

        # self.neurite_features_dict = dict(
        #     (id_, self.get_neurom_feature_annotations({"id": id_}))
        #     for id_ in neurite_features_base["id"]
        # )

        print("Including the following neurite features:")
        for n in self.neurite_features_df.columns:
            print("\t", n)

    def run(self) -> EmbeddingPipeline:

        # filtered_out = self.frame._nodes.reset_index()["@id"]
        # filtered_out = filtered_out.loc[self.neurite_features_df["@id"]]
        # neurite_features = pd.concat([
        #     filtered_out,
        #     self.neurite_features_df
        # ], axis=1).set_index("@id")
        # print(len(self.neurite_features_df))
        # print(len(self.frame._nodes))

        neurite_frame = PandasPGFrame.from_frames(
            nodes=self.neurite_features_df, edges=pd.DataFrame()
        )

        for c in neurite_frame._nodes.columns:
            try:
                neurite_frame.node_prop_as_numeric(c)
            except Exception:
                neurite_frame.node_prop_as_category(c)

        encoder = ScikitLearnPGEncoder(
            node_properties=neurite_frame.node_properties(),
            missing_numeric="impute",
            imputation_strategy="mean"
        )

        encoded_frame = encoder.fit_transform(neurite_frame)

        neurite_features = encoded_frame._nodes.rename(
            columns={"features": "neurite_features"}
        )

        data = np.array(neurite_features["neurite_features"].tolist())

        neurite_features["neurite_features"] = (data / data.max()).tolist()

        neurite_dim = len(neurite_features["neurite_features"].iloc[0])
        self.dimension = neurite_dim

        similarity_index = FaissSimilarityIndex(
            dimension=neurite_dim, similarity=self.similarity, n_segments=3
        )

        sim_processor = SimilarityProcessor(similarity_index, point_ids=None)

        point_ids = neurite_features.index
        sim_processor.add(neurite_features["neurite_features"].tolist(), point_ids)

        pipeline = EmbeddingPipeline(
            preprocessor=encoder,
            embedder=None,
            similarity_processor=sim_processor
        )

        return pipeline

    def get_neurom_feature_annotations(self, data: pd.Series, model_data: NeuronMorphologies)\
            -> pd.Series:

        record = {"id": data["id"]}

        if len(model_data.annotations) == 0:
            return pd.Series(record)

        annotation_i = [
            ann for ann in model_data.annotations[data["id"]][1]
            if "NeuronMorphologyFeatureAnnotation" in ann.type and
               ann.compartment not in self.compartments_to_exclude
        ]

        for ann in annotation_i:

            non_location_features = [
                feature_ann for feature_ann in ann.hasBody
                if feature_ann.isMeasurementOf.label not in model_data.location_feature_names
            ]

            for feature_ann in non_location_features:

                record_i = dict(
                    (
                        f"{ann.compartment}_"
                        f"{el.statistics.replace(' ', '_')}_"
                        f"{feature_ann.isMeasurementOf.label.replace(' ', '_')}",
                        el.value
                    )
                    for el in feature_ann.value.series
                    if el.statistics in self.statistics_of_interest and "value" in
                    el.__dict__ and el.value is not None
                )
                record.update(record_i)

        return pd.Series(record)


if __name__ == "__main__":
    a = NeuronMorphologiesLoad(
        bucket_configuration=NexusBucketConfiguration(
            organisation="bbp-external", project="seu", deployment=Deployment.PRODUCTION
        ),
        save_dir="../model_data_impl/test"
    )
    e = NeuriteModel(a).run()
