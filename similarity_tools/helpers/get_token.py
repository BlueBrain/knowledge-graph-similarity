import requests
import base64
import os

from similarity_tools.helpers.bucket_configuration import Deployment


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
