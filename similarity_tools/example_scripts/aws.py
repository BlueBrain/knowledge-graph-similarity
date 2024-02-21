from inference_tools.similarity.formula import Formula
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment
from similarity_tools.building.model_data_impl.neuron_morphologies_query \
    import NeuronMorphologiesQuery

from similarity_tools.building.model_impl.tmd_model.tmd_model import UnscaledTMDModel
from similarity_tools.helpers.elastic import ElasticSearch
from similarity_tools.registration.model_registration_pipeline import ModelRegistrationPipeline
from similarity_tools.registration.step import Step
from similarity_tools.building.model_descriptions.model_desc_list_no_class import \
    unscaled_model_description

deployment = Deployment.AWS
org, project = "public", "hippocampus"

# ----------------------------------------------- 1 -----------------------------------------------

# model_data_nm = NeuronMorphologiesQuery(
#     org=org, project=project, deployment=deployment, save_dir="./temp",
#     get_annotations=False
# )
#
# unscaled_model_description.model = UnscaledTMDModel
#
# e = ModelRegistrationPipeline.get_step(Step.SAVE_MODEL).run(
#     model_description=unscaled_model_description, model_data=model_data_nm
# )
# print(e)

# OUTPUT PATH:
# "/Users/mouffok/work_dir/kg-inference/similarity_tools/../data/pipelines/morph_TMD_euclidean_public_hippocampus_aws.zip"

# ----------------------------------------------- 2 -----------------------------------------------

# unscaled_model_description.filename = "morph_TMD_euclidean_public_hippocampus_aws"
#
# e = ModelRegistrationPipeline.get_step(Step.REGISTER_MODEL).run(
#     model_description=unscaled_model_description,
#     model_bc=NexusBucketConfiguration(organisation=org, project=project, deployment=deployment)
# )
# print(e)

# OUTPUT Embedding model id
# "https://sbo-nexus-delta.shapes-registry.org/v1/resources/public/hippocampus/_/e9fef977-6c2e-4c8a-a902-ed5a905a5888"

# ----------------------------------------------- 3 -----------------------------------------------

# e = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDING_MODEL_CATALOG).run(
#     model_name=unscaled_model_description.name,
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     bucket_list_rev=[
#         (NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment), None)
#     ],
#     target_type="NeuronMorphology"
# )
#
# print(e)

# Output Embedding Model data catalog id
# "https://sbo-nexus-delta.shapes-registry.org/v1/resources/bbp/atlas/_/068aba0c-7550-48c7-a24e-9b113263db00"

# ----------------------------------------------- 4 -----------------------------------------------

# AT STEP REGISTER EMBEDDINGS THIS IS WHERE THE TAG CREATION HAPPENS AND IT PROPAGATED AT FURTHER
# STEPS. IF TAG FORMAT SHOULD BE UPDATED, it's here
#
# embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
#     model_bc=NexusBucketConfiguration(organisation=org, project=project, deployment=deployment),
#     model_description=unscaled_model_description
# )

# ----------------------------------------------- 5 -----------------------------------------------

embedding_tag = "e9fef977-6c2e-4c8a-a902-ed5a905a5888?rev=1"
vector_dimension = 256

view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_SIMILARITY_VIEW).run(
    bucket_configuration=NexusBucketConfiguration(
        organisation=org, project=project, deployment=deployment
    ),
    resource_tag=embedding_tag,
    vector_dimension=vector_dimension
)

# OUTPUT Similarity view id
# "https://staging.nise.bbp.epfl.ch/nexus/v1/views/public/hippocampus/b291a3da-6379-4fd9-a590-c2c7530aa967"

# ----------------------------------------------- 6 -----------------------------------------------

# view = "https://staging.nise.bbp.epfl.ch/nexus/v1/views/public/hippocampus/b291a3da-6379-4fd9-a590-c2c7530aa967"
# t = NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment)
#
# ag_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_AGGREGATED_SIMILARITY_VIEW).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     to_aggregate=[(t, view)]
# )

# OUTPUT Aggregated Similarity view id
# 'https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/1522b20e-f5c7-463a-86a0-cd033ce4a1df'

# ----------------------------------------------- 7 -----------------------------------------------

# score_formula = Formula(unscaled_model_description.distance)
# embedding_tag = "4897a90e-539d-4b7f-8288-672bbe2d65e4?rev=2"
#
# agg_sim_view_id = 'https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/1522b20e-f5c7-463a-86a0-cd033ce4a1df'
#
# stats_id = ModelRegistrationPipeline.get_step(Step.REGISTER_NON_BOOSTED_STATS).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     similarity_aggregated_view_id=agg_sim_view_id,
#     stats_tag=embedding_tag,
#     score_formula=score_formula,
#     derivation_type="NeuronMorphology"
# )

# OUTPUT Stats id
# "https://staging.nise.bbp.epfl.ch/nexus/v1/resources/bbp/atlas/_/2f026399-d402-477d-8872-a3e29daebb32"

# ----------------------------------------------- 8 -----------------------------------------------

# score_formula = Formula(unscaled_model_description.distance)
# embedding_tag = "4897a90e-539d-4b7f-8288-672bbe2d65e4?rev=2"
#
# agg_sim_view_id = 'https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/1522b20e-f5c7-463a-86a0-cd033ce4a1df'
# non_boosted_stat_id = "https://staging.nise.bbp.epfl.ch/nexus/v1/resources/bbp/atlas/_/2f026399-d402-477d-8872-a3e29daebb32"
#
# boosting_tag = ModelRegistrationPipeline.get_step(Step.REGISTER_BOOSTING_FACTORS).run(
#     aggregated_similarity_view_id=agg_sim_view_id,
#     non_boosted_stats_id=non_boosted_stat_id,
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     bucket_bc=NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment),
#     boosting_tag=embedding_tag,
#     score_formula=score_formula
# )

# ----------------------------------------------- 9 -----------------------------------------------

# boosting_tag = "4897a90e-539d-4b7f-8288-672bbe2d65e4?rev=2"
#
# boosting_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_BOOSTING_VIEW).run(
#     bucket_configuration=NexusBucketConfiguration(organisation="public", project="hippocampus",
#     deployment=deployment),
#     boosting_tag=boosting_tag
# )

# OUTPUT Boosting view:
# https://staging.nise.bbp.epfl.ch/nexus/v1/views/public/hippocampus/a89971fa-0fa9-4cef-ab35-acbaead5eab7

# ----------------------------------------------- 10 -----------------------------------------------

# view = "https://staging.nise.bbp.epfl.ch/nexus/v1/views/public/hippocampus/a89971fa-0fa9-4cef-ab35-acbaead5eab7"
# t = NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment)
#
# ag_boosting_view_id = ModelRegistrationPipeline.get_step(
#     Step.REGISTER_AGGREGATED_BOOSTING_VIEW).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     to_aggregate=[(t, view)]
# )

# OUTPUT Aggregated Boosting View
# "https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/1321392d-021a-4219-954f-64a4d49f3357"

# ----------------------------------------------- 11 -----------------------------------------------

# score_formula = Formula(unscaled_model_description.distance)
# embedding_tag = "4897a90e-539d-4b7f-8288-672bbe2d65e4?rev=2"
# aggregated_boosting_view = "https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/1321392d-021a-4219-954f-64a4d49f3357"
# agg_sim_view_id = 'https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/1522b20e-f5c7-463a-86a0-cd033ce4a1df'
#
# stats_id = ModelRegistrationPipeline.get_step(Step.REGISTER_BOOSTED_STATS).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     similarity_aggregated_view_id=agg_sim_view_id,
#     stats_tag=embedding_tag,
#     boosting_aggregated_view_id=aggregated_boosting_view,
#     score_formula=score_formula,
#     derivation_type="NeuronMorphology"
# )

# OUTPUT Stats id
# https://staging.nise.bbp.epfl.ch/nexus/v1/resources/bbp/atlas/_/3c9e21ad-a56c-44a1-bb3d-6f42c0715183

# ----------------------------------------------- 12 -----------------------------------------------

# embedding_tag = "4897a90e-539d-4b7f-8288-672bbe2d65e4?rev=2"
#
# boosting_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_STATS_VIEW).run(
#     bucket_configuration=NexusBucketConfiguration(organisation="bbp", project="atlas",
#                                                   deployment=deployment),
#     stats_tag=embedding_tag
# )

# OUTPUT Stats view
# https://staging.nise.bbp.epfl.ch/nexus/v1/views/bbp/atlas/b7241501-172f-4e0a-a4b4-61ea45d97231
