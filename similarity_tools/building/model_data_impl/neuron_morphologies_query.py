from typing import Dict, Tuple, List
import os
import json
import pandas as pd

from datetime import datetime
from kgforge.core.wrappings import FilterOperator, Filter

from kgforge.core import Resource

from similarity_tools.building.model_data_impl.neuron_morphologies import NeuronMorphologies
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.helpers.utils import encode_id_rev


class NeuronMorphologiesQuery(NeuronMorphologies):

    LIMIT = 10000

    def __init__(
            self,
            bucket_configuration: NexusBucketConfiguration,
            src_data_dir=None,
            dst_data_dir=None,
            save_dir=None,
            get_annotations=True,
            limit=LIMIT
    ):

        super().__init__(
            bucket_configuration=bucket_configuration,
            src_data_dir=src_data_dir,
            dst_data_dir=dst_data_dir,
        )

        filter_ = {
            "type": "ReconstructedNeuronMorphology",
            "annotation": {
                "hasBody": {
                    "label": "Curated"
                }
            }
        }

        self.forge = bucket_configuration.allocate_forge_session()

        morphologies = self.forge.search(filter_, limit=limit)

        self.data: List[Resource] = morphologies

        # self.save_dir = save_dir
        #
        # full_df: pd.DataFrame = self.forge.as_dataframe(morphologies, store_metadata=True)
        # # Save
        # if self.save_dir:
        #     os.makedirs(self.save_dir, exist_ok=True)
        #     full_df.to_pickle(os.path.join(self.save_dir, "morphologies.pkl"))
        #
        # full_df["id"] = full_df[["id", "_rev"]].apply(lambda x: encode_id_rev(x[0], x[1]), axis=1)
        # self.full_df: pd.DataFrame = full_df
        #
        # if get_annotations:
        #     self.annotations: Dict[str, Tuple[str, List[Resource]]] = dict(
        #         (
        #
        #             encode_id_rev(r.id, r._store_metadata._rev),
        #             (r.name, self.forge.search({
        #                 "type": "NeuronMorphologyFeatureAnnotation",
        #                 "hasTarget": {"hasSource": {"id": r.id}}
        #             }))
        #         )
        #         for r in morphologies
        #     )
        #
        #     # Save
        #     for id_, (name, ann) in self.annotations.items():
        #         df: pd.DataFrame = self.forge.as_dataframe(ann, store_metadata=True)
        #         df.to_pickle(os.path.join(self.save_dir, f"{name}.pkl"))
        #
        # else:
        #     self.annotations = dict()
        #
        # self.brain_region_notation = self._get_brain_region_notations(full_df)
        # self.morphologies_br_df = self.brain_region_dataframe(full_df)
        #
        # # Save
        # if self.save_dir:
        #     self.morphologies_br_df.to_pickle(os.path.join(self.save_dir, "morphologies_br_df.pkl"))
        #
        #     with open(os.path.join(self.save_dir, "brain_region_notation.json"), "w") as f:
        #         f.write(json.dumps(self.brain_region_notation))
        #
        # if self.annotations:
        #     self.location_features = pd.concat([
        #         self.morphologies_br_df,
        #         full_df.apply(self.get_location_feature_annotations, axis=1).set_index("id")
        #     ], axis=1)


if __name__ == "__main__":
    test = NexusBucketConfiguration(
        organisation="bbp-external", project="seu", deployment=Deployment.PRODUCTION
    )
    a = NeuronMorphologiesQuery(bucket_configuration=test, save_dir="./test")

# filter_1 = Filter(
#     operator=FilterOperator.EQUAL.value,
#     path=["type"],
#     value="ReconstructedNeuronMorphology"
# )
# filter_2 = Filter(
#     operator=FilterOperator.LOWER_OR_Equal_Than.value,
#     path=["_createdAt"],
#     value="2023-09-26T09:00:00.000Z" + "^^xsd:dateTime"
# )
#
# # the time around when Pati registered new morphologies for which annotations
# # shouldn't be created yet
#
# morphologies = self.forge.search(filter_1, filter_2, limit=1500)
