"""Utils for registering similarity-related resources in Nexus."""
from typing import Tuple, Optional
from urllib import parse
import os

from kgforge.core import KnowledgeGraphForge, Resource
import uuid

from inference_tools.nexus_utils.forge_utils import ForgeUtils
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import SimilarityToolsException


def raise_error_on_failure(resource: Resource):
    last_action = resource._last_action
    logger.info(f">  Success: {last_action.succeeded}")

    if last_action.error is not None:
        logger.error(last_action.error)
        raise SimilarityToolsException(last_action.message)


def create_id_with_config(
        bucket_configuration: NexusBucketConfiguration, post_str: Optional[str] = None,
        is_view=False
) -> str:
    return create_id(
        org=bucket_configuration.organisation,
        project=bucket_configuration.project,
        endpoint=bucket_configuration.endpoint,
        post_str=post_str, is_view=is_view
    )


def create_id_with_forge(
        forge: KnowledgeGraphForge,
        post_str: Optional[str] = None,
        is_view=False
) -> str:

    endpoint, org, project = ForgeUtils.get_endpoint_org_project(forge)
    return create_id(
        org=org, project=project, endpoint=endpoint, post_str=post_str, is_view=is_view
    )


def create_id(
        org: str, project: str, endpoint: str, post_str: Optional[str] = None,
        is_view=False
) -> str:
    post_str = f"{post_str}/" if post_str is not None else ""
    t = "resources" if not is_view else "views"
    post_str = "_/" + post_str if not is_view else post_str

    endpoint = endpoint.replace("nexus/v1", "")
    return os.path.join(endpoint, t, org, project, f"{post_str}{uuid.uuid4()}")


def encode_id_rev_resource(resource: Resource) -> str:
    return encode_id_rev(resource.id, resource._store_metadata._rev)


def encode_id_rev(resource_id: str, resource_rev: int) -> str:
    return f"{resource_id}?{parse.urlencode({'rev': resource_rev})}"


def parse_id_rev(resource_str: str) -> Tuple[str, Optional[int]]:
    s = resource_str.split('?', 1)
    rev = dict(parse.parse_qsl(s[1]))["rev"] if len(s) > 1 else None
    return s[0], int(rev)


def get_path(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", path)


def get_model_tag(model_id, model_revision):
    model_uuid = model_id.split('/')[-1]
    return f"{model_uuid}?rev={model_revision}"


def get_stat_view_id(model):
    return get_x_view_id("stat", model)


def get_boosting_view_id(model):
    return get_x_view_id("boosting", model)


def get_similarity_view_id(model):
    return get_x_view_id("similarity", model)


def get_boosting_aggregated_view_id(model):
    return get_x_view_id("boosting_aggregated", model)


def get_similarity_aggregated_view_id(model):
    return get_x_view_id("similarity_aggregated", model)


def get_x_view_id(x: str, model):
    name = model.prefLabel.replace("(", "").replace(")", "").replace(" ", "_").lower()
    view_name = f"{name}_{x}_view"
    return f"https://bbp.epfl.ch/neurosciencegraph/data/views/es/{view_name}"
