import os
import json
import pandas as pd
from similarity_tools.building.model_data_impl.neuron_morphologies import NeuronMorphologies
from similarity_tools.helpers.bucket_configuration import Deployment

from similarity_tools.helpers.utils import encode_id_rev


class NeuronMorphologiesLoad(NeuronMorphologies):

    def __init__(
            self, org, project, save_dir, src_data_dir=None, dst_data_dir=None,
            get_annotations=True, deployment=Deployment.PRODUCTION,
            token_file_path=None, config_file_path=None
    ):
        super().__init__(
            src_data_dir=src_data_dir, dst_data_dir=dst_data_dir, org=org, project=project,
            deployment=deployment, get_annotations=get_annotations, token_file_path=token_file_path,
            config_file_path=config_file_path, save_dir=save_dir
        )

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
    b = NeuronMorphologiesLoad(org="bbp-external", project="seu", save_dir="./test")
