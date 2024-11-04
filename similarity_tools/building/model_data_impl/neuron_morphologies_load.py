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

import os
import json
import pandas as pd
from similarity_tools.building.model_data_impl.neuron_morphologies import NeuronMorphologies
from similarity_tools.helpers.bucket_configuration import Deployment, NexusBucketConfiguration

from similarity_tools.helpers.utils import encode_id_rev


class NeuronMorphologiesLoad(NeuronMorphologies):

    def __init__(
            self,
            bucket_configuration: NexusBucketConfiguration,
            save_dir,
            src_data_dir=None,
            dst_data_dir=None,
            get_annotations=True
    ):
        super().__init__(
            bucket_configuration=bucket_configuration,
            src_data_dir=src_data_dir, dst_data_dir=dst_data_dir,
        )

        self.forge = bucket_configuration.allocate_forge_session()

        full_df = pd.read_pickle(os.path.join(save_dir, "morphologies.pkl"))
        morphologies = self.forge.from_dataframe(full_df)
        self.data = morphologies

        full_df["id"] = full_df[["id", "_rev"]].apply(lambda x: encode_id_rev(x[0], x[1]), axis=1)
        self.full_df: pd.DataFrame = full_df

        self.morphologies_br_df = pd.read_pickle(os.path.join(save_dir, "morphologies_br_df.pkl"))

        with open(os.path.join(save_dir, "brain_region_notation.json"), "r") as f:
            self.brain_region_notation = json.loads(f.read())

        if get_annotations:
            # Load
            self.annotations = dict(
                (
                    encode_id_rev(
                        m.id, m._store_metadata._rev if m._store_metadata is not None else m._rev
                    ),
                    (m.name, self.forge.from_dataframe(
                        pd.read_pickle(os.path.join(save_dir, f"{m.name}.pkl"))
                    ))
                )
                for m in self.data
            )

            self.location_features = pd.concat([
                self.morphologies_br_df,
                full_df.apply(self.get_location_feature_annotations, axis=1).set_index("id")
            ], axis=1)
        else:
            self.annotations = dict()


if __name__ == "__main__":
    test = NexusBucketConfiguration(
        organisation="bbp-external", project="seu", deployment=Deployment.PRODUCTION
    )
    b = NeuronMorphologiesLoad(bucket_configuration=test, save_dir="./test")
