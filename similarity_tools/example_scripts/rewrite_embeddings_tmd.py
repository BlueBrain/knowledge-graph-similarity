from similarity_tools.building.model_data_impl.neuron_morphologies_query import \
    NeuronMorphologiesQuery
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.registration.model_registration_pipeline import ModelRegistrationPipeline
from similarity_tools.registration.step import Step
from similarity_tools.building.model_descriptions.model_desc_list_no_class import \
    unscaled_model_description

# ----------------------------------------------- 4 -----------------------------------------------

for org, project in [
    # ("public", "hippocampus"),
    # ("bbp-external", "seu"),
    ("public", "thalamus"),
    # ("bbp", "mouselight")
]:

    deployment = Deployment.PRODUCTION

    print(org, project)
    e = NeuronMorphologiesQuery(
        org, project, get_annotations=False, deployment=deployment
    )

    resource_id_rev_list = [(el.id, None) for el in e.data]

    print(len(resource_id_rev_list))

    embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
        model_bc=NexusBucketConfiguration(organisation=org, project=project, deployment=deployment),
        model_description=unscaled_model_description,
        # entity_type="NeuronMorphology",
        embedding_tag_transformer=lambda x: x + "_2",
        resource_id_rev_list=resource_id_rev_list
        # there are already embeddings pushed for the latest revision of the specified model,
        # in order not to overwrite them, this new version will be tagged by another tag,
        # so that these embeddings are not pushed directly to production (production is linked to
        # the default tag, because the views in the rule body point to this default tag)
    )

# ----------------------------------------------- 5 -----------------------------------------------


# for org, project, embedding_tag, vector_dimension in [
#     ("public", "hippocampus", "6f4331c1-ff9a-49e9-8957-f0372c82168d?rev=2_2", 256),
#     ("bbp-external", "seu", "9ddab7e4-f0b3-408a-b425-c1b6827f9ba0?rev=3_2", 256),
#     ("public", "thalamus", "d223feaa-a132-4547-a82b-d9c3147fb9b9?rev=2_2", 256),
#     ("bbp", "mouselight", "6a843c5e-a654-4a03-90d1-84da05949d19?rev=2_2", 256)
# ]:
#
#     view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_SIMILARITY_VIEW).run(
#         bucket_configuration=NexusBucketConfiguration(
#             organisation=org, project=project, deployment=Deployment.PRODUCTION
#         ),
#         resource_tag=embedding_tag,
#         vector_dimension=vector_dimension
#     )

# ----------------------------------------------- 6 -----------------------------------------------

# projects = [
#     ("public", "hippocampus", "https://bbp.epfl.ch/views/public/hippocampus/06c1ebcf-5b67-4888-b5de-6907e086fb81"),
#     ("bbp-external", "seu", "https://bbp.epfl.ch/views/bbp-external/seu/7c30af4a-4140-4a0b-ab19-eb84562a78a7"),
#     ("public", "thalamus", "https://bbp.epfl.ch/views/public/thalamus/67f07bd0-8eb8-4e76-9d79-0f87bcdd4325"),
#     ("bbp", "mouselight", "https://bbp.epfl.ch/views/bbp/mouselight/86a54def-f5fb-4596-96c7-a1cbf92acea0")
# ]
#
# to_aggregate = [
#     (
#         NexusBucketConfiguration(organisation=org, project=project, deployment=Deployment.PRODUCTION),
#         similarity_view_id
#     )
#     for org, project, similarity_view_id in projects
# ]
#
# ag_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_AGGREGATED_SIMILARITY_VIEW).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=Deployment.PRODUCTION),
#     to_aggregate=to_aggregate
# )

# OUTPUT Aggregated Similarity view id
# 'https://bbp.epfl.ch/views/bbp/atlas/32e5d0d4-b64b-4650-8a55-4b373bf75ea6'
