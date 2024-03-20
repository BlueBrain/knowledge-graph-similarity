from enum import Enum
from typing import Callable

from kgforge.core import KnowledgeGraphForge


import requests
import base64
import os


class Deployment(Enum):
    STAGING = "https://staging.nise.bbp.epfl.ch/nexus/v1"
    PRODUCTION = "https://bbp.epfl.ch/nexus/v1"
    AWS = "https://sbo-nexus-delta.shapes-registry.org/v1"


def get_path(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", path)


def auth(username, password, realm, server_url) -> requests.Response:

    def basic_auth():
        token = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
        return f'Basic {token}'

    url = f"{server_url}/realms/{realm}/protocol/openid-connect/token"

    resp = requests.post(
        url=url,
        headers={
            'Content-Type': "application/x-www-form-urlencoded",
            'Authorization': basic_auth()
        },
        data={
            'grant_type': "client_credentials",
            'scope': "openid"
        }
    )

    return resp.json()


def get_token_from_file(deployment: Deployment):

    def load_token(token_file_path: str):
        with open(token_file_path, encoding="utf-8") as f:
            return f.read()

    token_map = {
        Deployment.STAGING: get_path("../token/token_staging.txt"),
        Deployment.PRODUCTION: get_path("../token/token_prod.txt"),
        Deployment.AWS: get_path("../token/token_aws.txt")
    }

    return load_token(token_map[deployment])


# Wrapper around this can be used to provide username and password and
# then provide the callable to NexusBucketConfiguration
def _get_token(deployment: Deployment, username: str, password: str):

    if deployment == Deployment.AWS:
        server_url = "https://sboauth.epfl.ch/auth"
        realm = "SBO"
    else:
        server_url = "https://bbpauth.epfl.ch/auth/"
        realm = "BBP"

    resp = auth(username=username, password=password, server_url=server_url, realm=realm)
    return resp.json()["access_token"]



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
