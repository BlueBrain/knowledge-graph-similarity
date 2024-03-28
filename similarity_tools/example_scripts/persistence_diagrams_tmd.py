import os
import numpy as np
import copy
import tmd.Topology.distances

from similarity_tools.building.model_descriptions.model_desc_list_no_class import new_tmd_model_description
from similarity_tools.building.model_impl.tmd_model.persistence_diagram.persistence_diagram import \
    NeuriteType
from similarity_tools.building.model_impl.tmd_model.vectorisation import Vectorisation
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.building.model_impl.tmd_model.tmd_model import TMDModelNew, \
    VectorisationTechnique

from similarity_tools.building.model_data_impl.neuron_morphologies_query import \
    NeuronMorphologiesQuery

from tmd.Topology.vectorizations import get_limits

from similarity_tools.helpers.constants import DST_DATA_DIR, PIPELINE_SUBDIRECTORY
from similarity_tools.registration.model_registration_pipeline import ModelRegistrationPipeline
from similarity_tools.registration.step import Step

buckets = [
    ("public", "thalamus"),
    # ("bbp-external", "seu"),
    # ("bbp", "mouselight"),
    # ("public", "sscx"),
    # ("public", "hippocampus")
]

# new_tmd_model_description.filename = f"morph_TMD_image_data_base64_100"
#
# Vectorisation.BASE64 = True
#
# model_data_s = dict(
#     (
#         (org, project),
#         NeuronMorphologiesQuery(
#             bucket_configuration=NexusBucketConfiguration(
#                 organisation=org, project=project, deployment=Deployment.PRODUCTION
#             ),
#             get_annotations=False
#         )
#     )
#     for (org, project) in buckets
# )
#
# morphology_tmd_models = dict(
#     (
#         (org, project),
#         TMDModelNew(
#             model_data=model_data,
#             re_compute=False,
#             re_download=False,
#             neurite_type=NeuriteType.BASAL_DENDRITE
#         )
#     )
#     for (org, project), model_data in model_data_s.items()
# )
#
# list_of_ph = [
#     e
#     for morphology_tmd in morphology_tmd_models.values()
#     for e in list(morphology_tmd.nm_persistence_diagrams.values())
# ]
#
# xlim, ylim = get_limits(list_of_ph)
#
# diagrams = dict(
#     (
#         (org, project),
#         TMDModelNew.run_static(
#             vectorisation_technique=VectorisationTechnique.PERSISTENCE_IMAGE_DATA,
#             nm_persistence_diagrams=morphology_tmd.nm_persistence_diagrams,
#             xlim=xlim,
#             ylim=ylim
#         )
#     )
#     for (org, project), morphology_tmd in morphology_tmd_models.items()
# )
#
# for bucket in buckets:
#     pipeline = diagrams[bucket]
#     model_data = model_data_s[bucket]
#     e = ModelRegistrationPipeline.get_step(Step.SAVE_MODEL).run(
#         model_description=new_tmd_model_description, model_data=model_data, pipeline=pipeline
#     )
#
# exit()

push_bc = NexusBucketConfiguration(
    organisation="SarahTest", project="PublicThalamusTest2", deployment=Deployment.STAGING,
)

fn = copy.deepcopy(new_tmd_model_description.filename)

for org, project in buckets:

    model_bc = NexusBucketConfiguration(
        organisation=org, project=project, deployment=Deployment.PRODUCTION
    )

    # e = ModelRegistrationPipeline.get_step(Step.REGISTER_MODEL).run(
    #     model_description=new_tmd_model_description,
    #     model_bc=model_bc,
    # )

    model_path = os.path.join(
        DST_DATA_DIR, PIPELINE_SUBDIRECTORY,
        f"morph_TMD_image_data_base64_100_{org}_{project}_production.json"
    )

    # embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
    #     data_bc=model_bc,
    #     push_bc=push_bc,
    #     model_path=model_path,
    #     tag="test_tag3"
    #     # model_description=new_tmd_model_description
    #     # model_bc=model_bc
    # )

    # view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_SIMILARITY_VIEW).run(
    #     bucket_configuration=push_bc,
    #     resource_tag="test_tag3",
    #     vector_dimension=53336
    # )

# e = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDING_MODEL_CATALOG).run(
#     model_name=new_tmd_model_description.name,
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     bucket_list_rev=[
#         (NexusBucketConfiguration(organisation=org, project=project, deployment=deployment), None)
#         for org, project in buckets
#     ],
#     target_type="NeuronMorphology"
# )
