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

from abc import ABC, abstractmethod

from kgforge.core import Resource, KnowledgeGraphForge

from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import ModelBuildingException
from similarity_tools.helpers.utils import encode_id_rev, encode_id_rev_resource

from typing import List, Optional, Dict
from enum import Enum


class NeuriteType(Enum):
    BASAL_DENDRITE = "basal_dendrite"
    APICAL_DENDRITE = "apical_dendrite"
    AXON = "axon"


class PersistenceDiagram(ABC):
    FILTRATION_METRIC = "path_distances"

    @classmethod
    @abstractmethod
    def get_persistence_data(cls, filename: str, neurite_type: NeuriteType) -> Optional[List]:
        pass

    @classmethod
    @abstractmethod
    def get_distribution(cls, m: Resource, forge: KnowledgeGraphForge) -> Optional[Resource]:
        pass

    @classmethod
    def get_distributions(
            cls, data: List[Resource], download_dir: str, forge: KnowledgeGraphForge, download: bool
    ) -> Dict[str, str]:

        if (data is None or forge is None) and download_dir:
            raise ModelBuildingException("Missing data or forge instance, cannot re-download")

        distribution_resources: Dict[str, Resource] = dict(
            (encode_id_rev_resource(m), cls.get_distribution(m, forge))
            for m in data
        )

        if not download:
            logger.info("1. Getting local files")
        else:
            logger.info("1. Download content url")
            PersistenceDiagram._download_distribution(
                forge=forge, data=data, download_dir=download_dir,
                distribution_resources=distribution_resources
            )

        logger.info("2. Load files")

        return dict(
            (
                id_rev,
                f"{PersistenceDiagram._distribution_path(id_rev, download_dir)}/{distribution.name}"
            )
            for id_rev, distribution in distribution_resources.items()
        )

    @staticmethod
    def _distribution_path(id_rev: str, download_dir: str):
        # filename = d.atLocation.location.split('/')[-1]
        uuid_rev = id_rev.split('/')[-1]
        return f"{download_dir}/{uuid_rev}"

    @staticmethod
    def _download_distribution(
            forge: KnowledgeGraphForge, data: List[Resource], download_dir: str,
            distribution_resources: Dict[str, Resource]
    ):
        logger.info(f"Downloading {len(data)} entities to {download_dir}'...")

        os.makedirs(os.path.dirname(download_dir), exist_ok=True)

        for m in data:
            id_rev = encode_id_rev_resource(m)
            path = PersistenceDiagram._distribution_path(id_rev, download_dir)
            d = distribution_resources[id_rev]

            if d is not None:
                forge.download(d, "contentUrl", path=path, overwrite=True)
            else:
                logger.info(f">  Missing file for {m.name}")

        logger.info(">  Finished downloading files")

    @classmethod
    def recompute_persistence_diagrams(
            cls,
            download_dir: str,
            forge: KnowledgeGraphForge,
            persistence_diagram_location: str,
            data: List[Resource],
            re_download: bool,
            neurite_type: NeuriteType
    ):

        id_to_filename: Dict[str, str] = cls.get_distributions(
            data=data, forge=forge, download_dir=download_dir, download=re_download
        )

        computation: Dict[str, Optional[List]] = dict(
            (k, cls.get_persistence_data(v, neurite_type)) for k, v in id_to_filename.items()
        )

        diagrams: Dict[str, List] = dict(
            (k, v) for k, v in computation.items() if v is not None
        )

        os.makedirs(os.path.dirname(persistence_diagram_location), exist_ok=True)

        with open(persistence_diagram_location, "w") as f:
            json.dump(diagrams, f)

        return diagrams

    @classmethod
    def get_persistence_diagrams(
            cls,
            re_download: bool,
            re_compute: bool,
            forge: KnowledgeGraphForge,
            data: Optional[List[Resource]],
            persistence_diagram_location: str,
            download_dir: str,
            neurite_type: NeuriteType
    ) -> Dict[str, List]:

        if (data is None or forge is None) and re_compute:
            raise ModelBuildingException("Missing data or forge instance, cannot recompute")

        if re_compute:
            diagrams = cls.recompute_persistence_diagrams(
                persistence_diagram_location=persistence_diagram_location,
                download_dir=download_dir,
                forge=forge,
                data=data,
                re_download=re_download,
                neurite_type=neurite_type
            )
        else:
            with open(persistence_diagram_location, "r") as f:
                diagrams = json.load(f)

        return diagrams
