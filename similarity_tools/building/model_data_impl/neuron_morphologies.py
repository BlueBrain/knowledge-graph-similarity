from typing import Dict, Tuple, List, Set
import pandas as pd

from kgforge.core import Resource

from similarity_tools.helpers.elastic import ElasticSearch

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment
from similarity_tools.data_classes.model_data import ModelData


class NeuronMorphologies(ModelData):
    location_feature_names = ["Section Regions", "Leaf Regions"]
    compartments_with_location_features = ["Axon", "BasalDendrite", "ApicalDendrite"]
    dke_metrics = location_feature_names + ["Soma Number Of Points"]
    compartments_to_exclude = []
    annotations: Dict[str, Tuple[str, List[Resource]]]
    brain_region_notation: Dict
    location_features: pd.DataFrame
    full_df: pd.DataFrame
    morphologies_br_df: pd.DataFrame

    def __init__(
            self, org, project, src_data_dir=None, dst_data_dir=None,
            get_annotations=True,
            deployment=Deployment.PRODUCTION, config_file_path=None
    ):

        super().__init__(
            src_data_dir=src_data_dir, dst_data_dir=dst_data_dir, org=org, project=project,
            deployment=deployment
        )

        bucket_configuration = NexusBucketConfiguration(
            self.org, self.project, deployment=deployment,
            config_file_path=config_file_path,
            sparql_view="https://bluebrain.github.io/nexus/vocabulary/defaultSparqlIndex"
        )

        self.forge = bucket_configuration.allocate_forge_session()

    def brain_region_dataframe(self, full_df) -> pd.DataFrame:

        morphologies_br_df: pd.DataFrame = full_df[[
            "id",
            "brainLocation.brainRegion.id"
        ]]

        # Replace id with notation
        morphologies_br_df.loc[:, 'brainLocation.brainRegion.id'] = \
            morphologies_br_df["brainLocation.brainRegion.id"].apply(
                lambda x: self.brain_region_notation.get(x, x)
            )

        morphologies_br_df = morphologies_br_df.set_index("id")

        return morphologies_br_df

    def _get_brain_region_notations(self, full_df) -> Dict[str, str]:

        brain_region_ids = full_df["brainLocation.brainRegion.id"].to_list()
        all_ids = full_df["id"].to_list()

        br_keys = list(map(lambda id_: self._get_missing_brain_region_notations(
            id_=id_,
            existing_br_notations=brain_region_ids
        ), all_ids))

        br_keys = list(set.union(*br_keys)) + brain_region_ids

        forge_datamodels = NexusBucketConfiguration(
            "neurosciencegraph", "datamodels", deployment=self.deployment,
            elastic_search_view="https://bbp.epfl.ch/neurosciencegraph/data/views/es/dataset"
        ).allocate_forge_session()

        return {
            r.id: r.notation  # (r.notation, r.prefLabel)
            for r in ElasticSearch.get_by_ids(br_keys, forge_datamodels)
        }

    def _get_missing_brain_region_notations(
            self, id_: str, existing_br_notations: List[str]
    ) -> Set[str]:

        if len(self.annotations) == 0:
            return set()

        def get_br_in_annotation(annotation):
            location_features = [
                feature_ann for feature_ann in annotation.hasBody
                if feature_ann.isMeasurementOf.label in NeuronMorphologies.location_feature_names
            ]

            def br_id(series_element):
                return series_element.brainRegion.id

            return [
                br_id(series_item)
                for feature_ann in location_features
                for series_item in feature_ann.value.series
                if br_id(series_item) is not None
            ]

        valid_annotation = [
            ann for ann in self.annotations[id_][1]
            if "NeuronMorphologyFeatureAnnotation" in ann.type
               and ann.compartment not in self.compartments_to_exclude
        ]

        return {
            e
            for ann in valid_annotation
            for e in get_br_in_annotation(ann)
            if e not in existing_br_notations
        }

    def get_location_feature_annotations(self, data: pd.Series) -> pd.Series:

        annotation_i = [
            ann for ann in self.annotations[data.id][1]
            if "NeuronMorphologyFeatureAnnotation" in ann.type and
               ann.compartment not in self.compartments_to_exclude
        ]

        record = {"id": data.id}

        for ann in annotation_i:

            record_i = dict(
                (
                    self.compartment_feature_name_to_str(
                        ann.compartment, feature_ann.isMeasurementOf.label
                    ),
                    [
                        e for r in feature_ann.value.series for e in
                        [self.brain_region_notation[r.brainRegion.id]] * r.count
                    ]
                )
                for feature_ann in ann.hasBody
                if feature_ann.isMeasurementOf.label in
                NeuronMorphologies.location_feature_names
            )

            record.update(record_i)

        return pd.Series(record)

    def compartment_feature_name_to_str(self, compartment, feature_name):
        return f"{compartment}_{feature_name.replace(' ', '_')}"
