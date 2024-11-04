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

import json
from shutil import ReadError
from typing import Any, Optional, Dict

from kgforge.core import KnowledgeGraphForge, Resource
from kgforge.specializations.resources import Dataset

from bluegraph.core.embed.embedders import GraphElementEmbedder
from bluegraph.downstream import EmbeddingPipeline

from similarity_tools.registration.helper_functions.common import _fetch_one, add_contribution
from similarity_tools.registration.helper_functions.software_agents import get_wasAssociatedWith
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.types import Types
from similarity_tools.helpers.utils import create_id_with_forge


def create_model(
        forge: KnowledgeGraphForge,
        name: str, description: str,
        pref_label: str,
        pipeline_path: str,
        distance_metric: str,
        vector_dimension: int,
        content_type: str
) -> Resource:
    """
    Create a new embedding model resource
    @param forge:
    @type forge: KnowledgeGraphForge
    @param name:
    @type name: str
    @param description:
    @type description: str
    @param pref_label:
    @type pref_label: str
    @param pipeline_path:
    @type pipeline_path:
    @param distance_metric:
    @type distance_metric:
    @param vector_dimension:
    @type vector_dimension:
    @param content_type:
    @type content_type:
    @return: the id of the new model
    @rtype: str
    """

    model_resource = Dataset(forge, name=name, type="EmbeddingModel", description=description)
    model_schema = forge._model.schema_id(Types.EMBEDDING_MODEL.value)
    model_resource.id = create_id_with_forge(forge)
    model_resource.prefLabel = pref_label
    model_resource.similarity = distance_metric
    model_resource.vectorDimension = vector_dimension

    if pipeline_path is not None:
        model_resource.add_distribution(pipeline_path, content_type=content_type)

    e = forge.from_json({
        "activity": {
            "wasAssociatedWith": get_wasAssociatedWith(bluegraph=content_type == "application/octet-stream")
        }
    })
    model_resource.add_generation(e)

    model_resource = add_contribution(forge=forge, resource=model_resource)

    forge.register(model_resource, schema_id=model_schema)
    return model_resource


def update_model(
        forge: KnowledgeGraphForge,
        model_resource: Resource,
        new_pipeline_path: str,
        vector_dimension: int,
        content_type: str
) -> Resource:

    model_schema = forge._model.schema_id(Types.EMBEDDING_MODEL.value)

    # TODO should only update the distribution or also more?
    model_resource.vectorDimension = vector_dimension

    model_resource.distribution = forge.attach(new_pipeline_path, content_type=content_type)

    model_resource.generation = forge.from_json({
        "activity": {
            "wasAssociatedWith": get_wasAssociatedWith(
                bluegraph=content_type == "application/octet-stream"
            )
        }
    })

    forge.update(model_resource, schema_id=model_schema)
    return model_resource


def fetch_embedding_model_data_catalog(
        forge: KnowledgeGraphForge, data_catalog_name: str, data_catalog_rev: Optional[int] = None
) -> Optional[Resource]:

    return _fetch_one(
        entity_name=data_catalog_name,
        forge=forge,
        entity_rev=data_catalog_rev,
        entity_type=Types.EMBEDDING_MODEL_DATA_CATALOG.value,
        type_str="embedding model data catalog"
    )


def fetch_model(
        forge: KnowledgeGraphForge, model_name: str, model_rev: Optional[int] = None
) -> Optional[Resource]:
    return _fetch_one(
        entity_name=model_name,
        forge=forge,
        entity_rev=model_rev,
        entity_type=Types.EMBEDDING_MODEL.value,
        type_str="model"
    )


def push_model(
        forge: KnowledgeGraphForge,
        model_name: str,
        description: str,
        label: str,
        pipeline_path: str,
        distance_metric: str
) -> Resource:
    """
    Push (register or update) an embedding model
    @param forge:
    @type forge: KnowledgeGraphForge
    @param model_name:
    @type model_name: str
    @param description:
    @type description: str
    @param label:
    @type label: str
    @param pipeline_path:
    @type pipeline_path:
    @param distance_metric:
    @type distance_metric:
    @return: the updated/created model
    @rtype: Resource
    """
    try:
        path = f"{pipeline_path}.zip"
        pipeline: EmbeddingPipeline = EmbeddingPipeline.load(
            path=path, embedder_interface=GraphElementEmbedder, embedder_ext="zip"
        )
        vector_dimension = pipeline.generate_embedding_table().iloc[0]["embedding"].shape[0]
        content_type = "application/octet-stream"
        pipeline_path = path

    except ReadError:
        path = f"{pipeline_path}.json"
        with open(path, "r") as f:
            pipeline: Dict = json.load(f)

        vector_dimension = len(list(pipeline.values())[0])
        content_type = "application/json"
        pipeline_path = path

    existing_model = fetch_model(forge, model_name=model_name)

    if existing_model is not None:
        logger.info(">  Embedding Model exists, updating it")
        model_resource = update_model(
            forge=forge,
            model_resource=existing_model,
            new_pipeline_path=pipeline_path,
            vector_dimension=vector_dimension,
            content_type=content_type
        )
    else:
        logger.info(">  Embedding Model does not exist, creating it")
        model_resource = create_model(
            forge=forge,
            name=model_name,
            description=description,
            pref_label=label,
            pipeline_path=pipeline_path,
            distance_metric=distance_metric,
            vector_dimension=vector_dimension,
            content_type=content_type
        )
    return model_resource
