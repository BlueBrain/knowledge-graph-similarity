from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.registration.model_registration_pipeline import ModelRegistrationPipeline
from similarity_tools.registration.step import Step
from similarity_tools.building.model_descriptions.model_desc_list_no_class import \
    bbp_brain_region_bmo_model_description

deployment = Deployment.PRODUCTION

# ----------------------------------------------- 4 -----------------------------------------------

# for org, project in [
#     ("bbp-external", "seu"),
#     ("public", "thalamus"),
#     ("bbp", "mouselight")
# ]:
#
#     embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
#         model_bc=NexusBucketConfiguration(organisation=org, project=project, deployment=deployment),
#         model_description=bbp_brain_region_bmo_model_description,
#         entity_type="NeuronMorphology",
#         embedding_tag_transformer=lambda x: x + "_1"
#         # there are already embeddings pushed for the latest revision of the specified model,
#         # in order not to overwrite them, this new version will be tagged by another tag,
#         # so that these embeddings are not pushed directly to production (production is linked to
#         # the default tag, because the views in the rule body point to this default tag)
#     )

# ----------------------------------------------- 5 -----------------------------------------------

# for org, project, embedding_tag, vector_dimension in [
#     ("bbp-external", "seu", "35acc631-5bee-465b-9522-381a038180fc?rev=2_1", 32),
#     ("public", "thalamus", "192395fa-81fa-44e6-9409-e98643eb8ad7?rev=2_1", 32),
#     ("bbp", "mouselight", "d268edb7-91ed-4915-84cf-3b102b0946cb?rev=2_1", 32)
# ]:
#
#     view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_SIMILARITY_VIEW).run(
#         bucket_configuration=NexusBucketConfiguration(
#             organisation=org, project=project, deployment=deployment
#         ),
#         resource_tag=embedding_tag,
#         vector_dimension=vector_dimension
#     )

# ----------------------------------------------- 6 -----------------------------------------------

projects = [
    ("bbp-external", "seu", "https://bbp.epfl.ch/views/bbp-external/seu/959857aa-13f3-4610-aa97-136750d815ab"),
    ("public", "thalamus", "https://bbp.epfl.ch/views/public/thalamus/a48d8a54-bf0e-4c12-a545-1c177a957390"),
    ("bbp", "mouselight", "https://bbp.epfl.ch/views/bbp/mouselight/56ee4d47-ac63-40e6-80f5-423f2a180fc2")
]

to_aggregate = [
    (
        NexusBucketConfiguration(organisation=org, project=project, deployment=deployment),
        similarity_view_id
    )
    for org, project, similarity_view_id in projects
]

ag_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_AGGREGATED_SIMILARITY_VIEW).run(
    joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
    to_aggregate=to_aggregate
)

# OUTPUT Aggregated Similarity view id
# 'https://bbp.epfl.ch/views/bbp/atlas/df5cff33-79f9-49be-9f87-b2171e361a13'
