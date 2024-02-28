from enum import Enum
from typing import Callable

from kgforge.core import KnowledgeGraphForge

from similarity_tools.helpers.get_token import get_token_from_file


class Deployment(Enum):
    STAGING = "https://staging.nise.bbp.epfl.ch/nexus/v1"
    PRODUCTION = "https://bbp.epfl.ch/nexus/v1"
    AWS = "https://sbo-nexus-delta.shapes-registry.org/v1"


class NexusBucketConfiguration:

    config_prod_path = "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml"

    def __init__(
            self, organisation: str, project: str, deployment: Deployment,
            elastic_search_view: str = None, sparql_view: str = None,
            config_file_path: str = None,
            token_getter: Callable[[Deployment], str] = get_token_from_file
    ):

        self.deployment = deployment
        self.endpoint = deployment.value
        self.organisation = organisation
        self.project = project

        self.config_file_path = config_file_path
        self.token_getter = token_getter

        self.elastic_search_view = elastic_search_view
        self.sparql_view = sparql_view

    def get_token(self):
        return self.token_getter(self.deployment)

    def allocate_forge_session(self):

        bucket = f"{self.organisation}/{self.project}"

        args = dict(
            configuration=self.config_prod_path,
            endpoint=self.endpoint,
            token=self.get_token(),
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
        """
        Return a new instance of this class with the same attributes values as the self,
        except for the views, that will take the values provided as parameters
        :param elastic_search_view: the elastic search view new value
        :type elastic_search_view: str
        :param sparql_view: the sparql view new value
        :type sparql_view: str
        :return: the copied instance with the modified views
        :rtype: NexusBucketConfiguration
        """
        return NexusBucketConfiguration(
            organisation=self.organisation, project=self.project, deployment=self.deployment,
            config_file_path=self.config_file_path, token_getter=self.token_getter,
            sparql_view=sparql_view, elastic_search_view=elastic_search_view
        )

    def __repr__(self):
        return f" {self.organisation}/{self.project} in {self.deployment.name.lower()}"
