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

from typing import List, Optional, Tuple, Dict

from kgforge.core import Resource, KnowledgeGraphForge
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.registration.helper_functions.model import fetch_model, \
    fetch_embedding_model_data_catalog

from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import SimilarityToolsException
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import create_id_with_forge


def push_embedding_model_data_catalog(
        model_name: str, joint_forge: KnowledgeGraphForge,
        bucket_list: List[Tuple[NexusBucketConfiguration, Optional[int]]],
        target_type: str
) -> Resource:
    has_part = build_has_part(model_name=model_name, bucket_list=bucket_list)

    catalog_name = _build_catalog_name_from_model_name(model_name)

    embedding_model_data_catalog = fetch_embedding_model_data_catalog(
        data_catalog_name=catalog_name,
        forge=joint_forge
    )

    if embedding_model_data_catalog is None:
        logger.info(">  Embedding Model Data catalog does not exist, creating it")
        catalog_resource = _create_embedding_model_data_catalog(
            catalog_name=catalog_name,
            joint_forge=joint_forge,
            target_type=target_type,
            has_part=has_part
        )
    else:
        logger.info(">  Embedding Model Data catalog exists, updating it")
        catalog_resource = _update_embedding_model_data_catalog(
            joint_forge=joint_forge,
            has_part=has_part,
            catalog_resource=embedding_model_data_catalog
        )
    return catalog_resource


def _update_embedding_model_data_catalog(
        catalog_resource: Resource,
        has_part: List[Dict],
        joint_forge: KnowledgeGraphForge,
) -> Resource:
    catalog_resource.hasPart = has_part
    joint_forge.validate(catalog_resource, type_=Types.EMBEDDING_MODEL_DATA_CATALOG.value)
    joint_forge.update(catalog_resource)
    return catalog_resource


def _build_catalog_name_from_model_name(model_name: str):
    return f"Catalog of {model_name} models"


def _create_embedding_model_data_catalog(
        catalog_name: str,
        has_part: List[Dict],
        joint_forge: KnowledgeGraphForge,
        target_type: str
) -> Resource:
    dict_value = {
        "id": create_id_with_forge(joint_forge),
        "type": [Types.EMBEDDING_MODEL_DATA_CATALOG.value, "DataCatalog"],
        "about": Types.EMBEDDING_MODEL.value,
        "name": catalog_name,
        "prefLabel": catalog_name,
        "targetType": target_type,
        "hasPart": has_part
    }

    catalog_resource = joint_forge.from_json(dict_value)

    joint_forge.validate(catalog_resource, type_=Types.EMBEDDING_MODEL_DATA_CATALOG.value)
    joint_forge.register(catalog_resource)

    return catalog_resource


def build_has_part(
        model_name: str, bucket_list: List[Tuple[NexusBucketConfiguration, Optional[int]]]
) -> List[Dict]:
    def _get_model_from_bucket(
            bucket_configuration: NexusBucketConfiguration, rev: Optional[int]
    ) -> Tuple[str, int]:

        test = fetch_model(
            forge=bucket_configuration.allocate_forge_session(),
            model_name=model_name, model_rev=rev
        )
        if test is not None:
            return test.id, test._store_metadata._rev
        else:
            raise SimilarityToolsException(
                f"Could not find {model_name}"
                f" in {bucket_configuration.organisation}/{bucket_configuration.project}"
            )

    def _write_embedding_part(
            embedding_model_id, embedding_model_rev,
            bucket_configuration: NexusBucketConfiguration
    ) -> Dict:
        return {
            "@id": embedding_model_id,
            "@type": [Types.EMBEDDING_MODEL.value],
            "_rev": embedding_model_rev,
            "org": bucket_configuration.organisation,
            "project": bucket_configuration.project
        }

    logger.info(">  Fetching models to include as parts of the embedding model")

    model_id_revs: Dict[Tuple[str, int], NexusBucketConfiguration] = dict(
        (_get_model_from_bucket(bucket_configuration, rev), bucket_configuration)
        for bucket_configuration, rev in bucket_list
    )

    return [
        _write_embedding_part(id_, rev_, bucket_configuration)
        for (id_, rev_), bucket_configuration in model_id_revs.items()
    ]
