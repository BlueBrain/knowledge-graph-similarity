from enum import Enum

from kgforge.core import KnowledgeGraphForge

import os


def get_path(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", path)


class Deployment(Enum):
    STAGING = "https://staging.nise.bbp.epfl.ch/nexus/v1"
    PRODUCTION = "https://bbp.epfl.ch/nexus/v1"
    AWS = "https://sbo-nexus-delta.shapes-registry.org/v1"


class NexusBucketConfiguration:

    token_map = {
        Deployment.STAGING:  get_path("../token/token_staging.txt"),
        Deployment.PRODUCTION: get_path("../token/token_prod.txt"),
        Deployment.AWS: get_path("../token/token_aws.txt")
    }
    config_prod_path = \
       "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml"

    def __init__(self, organisation: str, project: str, deployment: Deployment,
                 elastic_search_view: str = None, sparql_view: str = None,
                 config_file_path: str = None, token_file_path: str = None):

        self.deployment = deployment
        self.endpoint = deployment.value
        self.organisation = organisation
        self.project = project

        self.config_file_path = config_file_path
        self.token_file_path = token_file_path

        self.token = None

        self.elastic_search_view = elastic_search_view
        self.sparql_view = sparql_view

    def set_token(self, token):
        self.token = token

    def get_token_path(self) -> str:
        return self.token_file_path or NexusBucketConfiguration.token_map[self.deployment]

    @staticmethod
    def load_token(token_file_path: str):
        with open(token_file_path, encoding="utf-8") as f:
            return f.read()

    def allocate_forge_session(self):

        bucket = f"{self.organisation}/{self.project}"

        token = self.token or NexusBucketConfiguration.load_token(self.get_token_path())

        args = dict(
            configuration=self.config_prod_path,
            endpoint=self.endpoint,
            token=token,
            bucket=bucket,
            debug=False
        )

        search_endpoints = {}

        if self.elastic_search_view is not None:
            search_endpoints["elastic"] = {"endpoint": self.elastic_search_view}

        if self.sparql_view is not None:
            search_endpoints["sparql"] = {"endpoint": self.sparql_view}

        if len(search_endpoints) > 0:
            args["searchendpoints"] = search_endpoints

        return KnowledgeGraphForge(**args)

    def copy_with_views(self, elastic_search_view: str = None, sparql_view: str = None):

        return NexusBucketConfiguration(
            organisation=self.organisation, project=self.project, deployment=self.deployment,
            config_file_path=self.config_file_path, token_file_path=self.token_file_path,
            sparql_view=sparql_view, elastic_search_view=elastic_search_view
        )

    def __repr__(self):
        return f" {self.organisation}/{self.project} in {self.deployment.name.lower()}"
