from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.registration.model_registration_pipeline import ModelRegistrationPipeline
from similarity_tools.registration.step import Step
from similarity_tools.building.model_descriptions.model_desc_list_no_class import \
    axon_model_description


deployment = Deployment.PRODUCTION

# ----------------------------------------------- 4 -----------------------------------------------

resource_id_rev_list = []

for org, project in [
    ("bbp-external", "seu")
]:

    embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
        model_bc=NexusBucketConfiguration(organisation=org, project=project, deployment=deployment),
        model_description=axon_model_description,
        # entity_type="NeuronMorphology",
        # resource_id_rev_list=resource_id_rev_list,
        embedding_tag_transformer=lambda x: x + "_2",

        # there are already embeddings pushed for the latest revision of the specified model,
        # in order not to overwrite them, this new version will be tagged by another tag,
        # so that these embeddings are not pushed directly to production (production is linked to
        # the default tag, because the views in the rule body point to this default tag)
    )

# ----------------------------------------------- 5 -----------------------------------------------


# for org, project, embedding_tag, vector_dimension in [
#     ("bbp-external", "seu", "fd5352b6-1e28-4c85-a408-de49044ae1b9?rev=5_1", 128),
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

# projects = [
#     ("bbp-external", "seu", "https://bbp.epfl.ch/views/bbp-external/seu/f41ee338-0d34-466b-96bc-4312d1d65165"),
# ]
#
# to_aggregate = [
#     (
#         NexusBucketConfiguration(organisation=org, project=project, deployment=deployment),
#         similarity_view_id
#     )
#     for org, project, similarity_view_id in projects
# ]
#
# ag_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_AGGREGATED_SIMILARITY_VIEW).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     to_aggregate=to_aggregate
# )

# OUTPUT Aggregated Similarity view id
# 'https://bbp.epfl.ch/views/bbp/atlas/90b2bae0-584e-4f7c-abaa-d1af5ad393e3'

