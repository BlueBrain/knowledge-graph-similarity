import time
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
from similarity_tools.building.model_impl.tmd_model.vectorisation import Vectorisation
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.building.model_impl.tmd_model.tmd_model import TMDModelNew, \
    VectorisationTechnique

from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import NeuriteType
    
from similarity_tools.building.model_descriptions.model_desc_list_no_class import new_tmd_model_description

from tmd.Topology.vectorizations import get_limits


deployment = Deployment.PRODUCTION
org, project = "public", "morphologies"

# ----------------------------------------------- 1 -----------------------------------------------
public_morphos = NexusBucketConfiguration(
        organisation=org, project=project, deployment=deployment
    )

# model_data_nm = NeuronMorphologiesQuery(bucket_configuration=public_morphos, save_dir="./temp",
#     get_annotations=False
# )

# new_model = TMDModelNew(
#             model_data=model_data_nm,
#             re_compute=False,
#             re_download=False,
#             neurite_type=NeuriteType.BASAL_DENDRITE
#         )

new_tmd_model_description.filename = f"morph_TMD_image_data_base64_100"
# new_tmd_model_description.model = new_model
#
Vectorisation.BASE64 = True

# # list_of_ph = [ [e for e in list(new_model.nm_persistence_diagrams.values())] ]
# list_of_ph = [
#     e
#     for morphology_tmd in [new_model] 
#     for e in list(morphology_tmd.nm_persistence_diagrams.values())
# ]

# xlim, ylim = get_limits(list_of_ph)

# pipeline = TMDModelNew.run_static(
#             vectorisation_technique=VectorisationTechnique.PERSISTENCE_IMAGE_DATA,
#             nm_persistence_diagrams=new_model.nm_persistence_diagrams,
#             xlim=xlim,
#             ylim=ylim
#         )

# e = ModelRegistrationPipeline.get_step(Step.SAVE_MODEL).run(
#     model_description=new_tmd_model_description, model_data=model_data_nm,
#     pipeline=pipeline
# )
# print(e)

# OUTPUT PATH:
# "/Users/mouffok/work_dir/kg-inference/similarity_tools/../data/pipelines/morph_TMD_euclidean_public_hippocampus_aws.zip"

# ----------------------------------------------- 2 -----------------------------------------------

new_tmd_model_description.filename = "morph_TMD_image_data_base64_100_public_morphologies_production"

# e = ModelRegistrationPipeline.get_step(Step.REGISTER_MODEL).run(
#     model_description=new_tmd_model_description,
#     model_bc=public_morphos
# )
# print(e)

# OUTPUT Embedding model id
# 'https://bbp.epfl.ch/resources/public/morphologies/_/48af2750-a51a-4c78-ac12-eeba20313805'

# ----------------------------------------------- 3 -----------------------------------------------

# e = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDING_MODEL_CATALOG).run(
#     model_name=new_tmd_model_description.name,
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     bucket_list_rev=[
#         (NexusBucketConfiguration(organisation="bbp-external", project="seu", deployment=deployment), None),
#         (NexusBucketConfiguration(organisation="bbp", project="mouselight", deployment=deployment), None),
#         (NexusBucketConfiguration(organisation="public", project="sscx", deployment=deployment), None),
#         (NexusBucketConfiguration(organisation="public", project="thalamus", deployment=deployment), None),
#         (NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment), None),
#         (NexusBucketConfiguration(organisation="public", project="morphologies", deployment=deployment), None)
#     ],
#     target_type="NeuronMorphology"
# )

# print(e)

# Output Embedding Model data catalog id
# 'https://bbp.epfl.ch/resources/bbp/atlas/_/37cddb53-5326-45a5-88da-9e809943b662'

# ----------------------------------------------- 4 -----------------------------------------------

# AT STEP REGISTER EMBEDDINGS THIS IS WHERE THE TAG CREATION HAPPENS AND IT PROPAGATED AT FURTHER
# STEPS. IF TAG FORMAT SHOULD BE UPDATED, it's here

# embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
#     data_bc=public_morphos,
#     model_bc=public_morphos,
#     push_bc=public_morphos,
#     model_description=new_tmd_model_description
# )

# print(embedding_tag)
# print(vector_dimension)
# ----------------------------------------------- 5 -----------------------------------------------

# embedding_tag = "48af2750-a51a-4c78-ac12-eeba20313805?rev=1"
# vector_dimension = 53336

# view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_SIMILARITY_VIEW).run(
#     bucket_configuration=public_morphos,
#     resource_tag=embedding_tag,
#     vector_dimension=vector_dimension
# )
# print(view_id)

# OUTPUT Similarity view id
# "https://bbp.epfl.ch/views/public/morphologies/69de97c8-71a5-4650-9d5f-d8de531416ae"

# ----------------------------------------------- 6 -----------------------------------------------

# to_aggregate = [
#     (
#         NexusBucketConfiguration(organisation="public", project="thalamus", deployment=deployment),
#         "https://bbp.epfl.ch/views/public/thalamus/5f808747-6b7c-423a-913f-873683986cb0"
#     ),
#     (
#         NexusBucketConfiguration(organisation="bbp-external", project="seu", deployment=deployment),
#         "https://bbp.epfl.ch/views/bbp-external/seu/4822f62b-2b4e-4424-9b23-3d7893653307"
#     ),
#     (
#         NexusBucketConfiguration(organisation="bbp", project="mouselight", deployment=deployment),
#         "https://bbp.epfl.ch/views/bbp/mouselight/5322b607-e1af-4f68-b070-f7de330272ea"
#     ),
#     (
#         NexusBucketConfiguration(organisation="public", project="sscx", deployment=deployment),
#         "https://bbp.epfl.ch/views/public/sscx/ff03eee9-deed-4912-b923-af0594fb22cc"
#     ),
#     (
#         NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment),
#         "https://bbp.epfl.ch/views/public/hippocampus/2b7550d5-0196-4fa6-911f-936f73d20882"
#     ),
#     (
#         NexusBucketConfiguration(organisation="public", project="morphologies", deployment=deployment),
#         "https://bbp.epfl.ch/views/public/morphologies/69de97c8-71a5-4650-9d5f-d8de531416ae"
#     )
# ]


# ag_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_AGGREGATED_SIMILARITY_VIEW).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     to_aggregate=to_aggregate
# )

# OUTPUT Aggregated Similarity view id
# "https://bbp.epfl.ch/views/bbp/atlas/a030e0bc-6e9e-4327-a41d-22110ace512a"

# ----------------------------------------------- 7 -----------------------------------------------

# score_formula = Formula(new_tmd_model_description.distance)
# embedding_tag = "48af2750-a51a-4c78-ac12-eeba20313805?rev=1"

# agg_sim_view_id = 'https://bbp.epfl.ch/views/bbp/atlas/a030e0bc-6e9e-4327-a41d-22110ace512a'

# stats_id = ModelRegistrationPipeline.get_step(Step.REGISTER_NON_BOOSTED_STATS).run(
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     similarity_aggregated_view_id=agg_sim_view_id,
#     stats_tag=embedding_tag,
#     score_formula=score_formula,
#     derivation_type="NeuronMorphology"
# )

# OUTPUT Stats id
# "https://bbp.epfl.ch/resources/bbp/atlas/_/6d87fd82-43cb-4ef1-9c9b-efdd57790660"

# ----------------------------------------------- 8 -----------------------------------------------

score_formula = Formula(new_tmd_model_description.distance)
embedding_tag = "48af2750-a51a-4c78-ac12-eeba20313805?rev=1"

agg_sim_view_id = 'https://bbp.epfl.ch/views/bbp/atlas/a030e0bc-6e9e-4327-a41d-22110ace512a'
non_boosted_stat_id = "https://bbp.epfl.ch/resources/bbp/atlas/_/6d87fd82-43cb-4ef1-9c9b-efdd57790660"

buckets = [
        NexusBucketConfiguration(organisation="public", project="thalamus", deployment=deployment),
        NexusBucketConfiguration(organisation="bbp-external", project="seu", deployment=deployment),
        NexusBucketConfiguration(organisation="bbp", project="mouselight", deployment=deployment),
        NexusBucketConfiguration(organisation="public", project="sscx", deployment=deployment),
        NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment),
        NexusBucketConfiguration(organisation="public", project="morphologies", deployment=deployment)
]

boosting_tags = []
boosting_view_ids = []
for bc in buckets:
    boosting_tag = ModelRegistrationPipeline.get_step(Step.REGISTER_BOOSTING_FACTORS).run(
        aggregated_similarity_view_id=agg_sim_view_id,
        non_boosted_stats_id=non_boosted_stat_id,
        joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
        bucket_bc=bc,
        boosting_tag=embedding_tag,
        score_formula=score_formula
    )
    print("boosting_tag: ", boosting_tag)
    boosting_tags.append(boosting_tag)

# ----------------------------------------------- 9 -----------------------------------------------

# boosting_tag = "4897a90e-539d-4b7f-8288-672bbe2d65e4?rev=2"
#
    time.sleep(60)
    boosting_view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_BOOSTING_VIEW).run(
        bucket_configuration=bc,
        deployment=deployment,
        boosting_tag=boosting_tag
    )
    
    print("boosting_view_id: ", boosting_view_id)
    boosting_view_ids.append((bc, boosting_view_id))
# OUTPUT Boosting view:
# https://staging.nise.bbp.epfl.ch/nexus/v1/views/public/hippocampus/a89971fa-0fa9-4cef-ab35-acbaead5eab7

# ----------------------------------------------- 10 -----------------------------------------------

# view = "https://staging.nise.bbp.epfl.ch/nexus/v1/views/public/hippocampus/a89971fa-0fa9-4cef-ab35-acbaead5eab7"
# t = NexusBucketConfiguration(organisation="public", project="hippocampus", deployment=deployment)
#

time.sleep(120)
ag_boosting_view_id = ModelRegistrationPipeline.get_step(
    Step.REGISTER_AGGREGATED_BOOSTING_VIEW).run(
    joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
    to_aggregate=boosting_view_ids
)

print(ag_boosting_view_id)

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
