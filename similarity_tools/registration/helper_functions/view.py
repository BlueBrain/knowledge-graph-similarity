from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus as url_encode
import requests
import json

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from inference_tools.nexus_utils.delta_utils import DeltaUtils


def create_es_view(
        bucket_configuration: NexusBucketConfiguration,
        token: str,
        es_view_id: str,
        pipeline: List[Dict],
        mapping: Optional[Dict] = None,
        resource_tag: str = None
) -> Dict:
    print("mapping: ")
    print(json.dumps(mapping, indent=4))
    payload = {
        "@id": es_view_id,
        "@type": ["View", "ElasticSearchView"],
        "pipeline": pipeline,
        "mapping": mapping if mapping is not None else {}
    }

    if resource_tag is not None:
        payload["resourceTag"] = resource_tag

    url = f"{bucket_configuration.endpoint}/views/{url_encode(bucket_configuration.organisation)}" \
          f"/{url_encode(bucket_configuration.project)}"

    return DeltaUtils.check_response(
       requests.post(url=url, headers=DeltaUtils.make_header(token), json=payload)
    )


def get_es_view(
        bucket_configuration: NexusBucketConfiguration,
        token: str,
        es_view_id: str,
        with_metadata: bool = True
) -> Dict:
    url = f"{bucket_configuration.endpoint}/views/{url_encode(bucket_configuration.organisation)}" \
          f"/{url_encode(bucket_configuration.project)}/{url_encode(es_view_id)}" \
          f"{'/source' if not with_metadata else ''}"

    return DeltaUtils.check_response(
        requests.get(url=url, headers=DeltaUtils.make_header(token))
    )


def update_es_view_resource_tag(
        bucket_configuration: NexusBucketConfiguration,
        token: str,
        es_view_id: str,
        view_body: Dict,
        resource_tag: str,
        rev: int
) -> Dict:
    url = f"{bucket_configuration.endpoint}/views/{url_encode(bucket_configuration.organisation)}" \
          f"/{url_encode(bucket_configuration.project)}/{url_encode(es_view_id)}?rev={rev}"

    original_payload_keys = ["@id", "@type", "mapping", "pipeline"]
    view_body = dict((k, view_body[k]) for k in original_payload_keys if k in view_body)
    view_body["resourceTag"] = resource_tag

    return DeltaUtils.check_response(
        requests.put(url=url, headers=DeltaUtils.make_header(token), json=view_body)
    )


def create_es_view_legacy_params(
        bucket_configuration: NexusBucketConfiguration,
        token: str,
        es_view_id: str,
        mapping: Optional[Dict] = None,
        resource_tag: str = None,
        resource_types: Optional[List] = None,
        resource_schemas: Optional[List] = None,
        select_predicates: Optional[List] = None,
        default_label_predicates: Optional[bool] = False,
        source_as_text: Optional[bool] = False,
        include_metadata: bool = True,
        filter_deprecated: bool = True,
        construct_query: Optional[str] = None
) -> Dict:
    # TODO enable users to specify priority order in the pipeline for each param
    def build_pipeline():

        pipeline = []

        if filter_deprecated:
            pipeline.append({"name": "filterDeprecated"})

        if resource_schemas:
            pipeline.append({
                "name": "filterBySchema",
                "config": {
                    "types": resource_schemas
                }
            })

        if not include_metadata:
            pipeline.append({"name": "discardMetadata"})

        if default_label_predicates:
            pipeline.append({"name": "defaultLabelPredicates"})

        if source_as_text:
            pipeline.append({"name": "sourceAsText"})

        if resource_types:
            pipeline.append({
                "name": "filterByType",
                "config": {"types": resource_types}
            })
        if construct_query is not None:
            pipeline.append({
                "name": "dataConstructQuery",
                "config": {"query": construct_query}
            })
        if select_predicates:
            pipeline.append({
                "name": "selectPredicates",
                "config": {"predicates": select_predicates}
            })

        return pipeline

    return create_es_view(
        bucket_configuration=bucket_configuration, pipeline=build_pipeline(),
        token=token, resource_tag=resource_tag, es_view_id=es_view_id,
        mapping=mapping
    )


def view_create(
        mapping: Dict,
        resource_types: List,
        view_id: str,
        resource_tag: str,
        bucket_configuration: NexusBucketConfiguration,
) -> Dict:

    token = bucket_configuration.get_token()

    return create_es_view_legacy_params(
        bucket_configuration=bucket_configuration,
        es_view_id=view_id,
        resource_types=resource_types,
        mapping=mapping,
        token=token,
        resource_schemas=None,
        resource_tag=resource_tag
    )


# def view_search_create_update(
#         mapping: Dict,
#         resource_types: List,
#         view_id: str,
#         bucket_configuration: NexusBucketConfiguration,
#         model: Optional[Resource] = None
# ) -> Dict:
#
#     model_id = model.id
#
#     model_revision = model._store_metadata._rev
#
#     resource_tag = get_model_tag(model_id, model_revision)
#
#     token = bucket_configuration.get_token()
#
#     try:
#         existing_view = get_es_view(
#             es_view_id=view_id,
#             token=token,
#             bucket_configuration=bucket_configuration
#         )
#     except DeltaException as e:
#         if e.status_code == 404:
#             existing_view = None
#         else:
#             raise e
#
#     if existing_view is not None:
#         logger.info("2. View exists, updating resource tag only")
#         rev = existing_view["_rev"]
#
#         updated_view = update_es_view_resource_tag(
#             bucket_configuration=bucket_configuration,
#             resource_tag=resource_tag,
#             es_view_id=view_id,
#             rev=rev,
#             token=token,
#             view_body=existing_view
#         )
#         return updated_view
#
#     else:
#         logger.info("2. View does not exist, registering the view")
#
#         created_view = create_es_view_legacy_params(
#             bucket_configuration=bucket_configuration,
#             es_view_id=view_id,
#             resource_types=resource_types,
#             mapping=mapping,
#             token=token,
#             resource_schemas=None,
#             resource_tag=resource_tag,
#         )
#         return created_view
#         # TODO status code check and Exception


def create_aggregated_view(
        bucket_configuration: NexusBucketConfiguration,
        projects_to_aggregate: List[Tuple[NexusBucketConfiguration, str]],
        aggregated_view_id: str
):
    # aggregated_view_template = "https://bbp.epfl.ch/neurosciencegraph/data/views/es" \
    #                            "/$MODEL_$VIEWTYPE_aggregated_view"
    #
    # sub_view_template = "https://bbp.epfl.ch/neurosciencegraph/data/views/es" \
    #                     "/$MODEL_$VIEWTYPE_view"
    #
    # aggregated_view_id = aggregated_view_template \
    #     .replace("$MODEL", model_str)\
    #     .replace("$VIEWTYPE", view_type)
    #
    # sub_view_id = sub_view_template.replace("$MODEL", model_str).replace("$VIEWTYPE", view_type)

    token = bucket_configuration.get_token()

    url = f"{bucket_configuration.endpoint}/views/{url_encode(bucket_configuration.organisation)}" \
          f"/{url_encode(bucket_configuration.project)}"

    payload = {
        "@id": aggregated_view_id,
        "@type": "AggregateElasticSearchView",
        "views": [
            {
                "project": f"{nbc.organisation}/{nbc.project}",
                "viewId": sub_view_id
            }
            for nbc, sub_view_id, in projects_to_aggregate
        ]
    }

    return DeltaUtils.check_response(
        requests.post(url=url, headers=DeltaUtils.make_header(token), json=payload)
    )
