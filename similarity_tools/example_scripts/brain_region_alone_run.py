# This file is part of knowledge-graph-similarity.
# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from similarity_tools.building.model_impl.brain_region_alone import BrModelData, \
    BBPBrainRegionModelAlone

from inference_tools.similarity.formula import Formula
from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment
from similarity_tools.registration.model_registration_pipeline import ModelRegistrationPipeline
from similarity_tools.registration.step import Step
from similarity_tools.building.model_descriptions.model_desc_list_no_class import \
    bbp_brain_region_alone_model_description


# deployment = Deployment.PRODUCTION

# ----------------------------------------------- 1 -----------------------------------------------

# bc = NexusBucketConfiguration(
#     organisation="neurosciencegraph", project="datamodels", deployment=deployment
# )
#
# bbp_brain_region_alone_model_description.model = BBPBrainRegionModelAlone
#
# e = ModelRegistrationPipeline.get_step(Step.SAVE_MODEL).run(
#     model_description=bbp_brain_region_alone_model_description,
#     model_data=BrModelData(bucket_configuration=bc)
#
# )

# print(e)

# OUTPUT PATH:
# "/Users/mouffok/work_dir/kg-inference/similarity_tools/../data/pipelines/brain_region_poincare_bbp_neurosciencegraph_datamodels.zip"

# ----------------------------------------------- 2 -----------------------------------------------

# bbp_brain_region_alone_model_description.filename = "brain_region_poincare_bbp_neurosciencegraph_datamodels"
#
# e = ModelRegistrationPipeline.get_step(Step.REGISTER_MODEL).run(
#     model_description=bbp_brain_region_alone_model_description,
#     model_bc=NexusBucketConfiguration(
#         organisation="neurosciencegraph", project="datamodels", deployment=deployment
#     )
#
#
# )
# print(e)

# OUTPUT Embedding model id
# "https://bbp.epfl.ch/resources/neurosciencegraph/datamodels/_/27fb56d6-15ad-4df4-a82e-d52f34e19600"

# ----------------------------------------------- 3 -----------------------------------------------

# e = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDING_MODEL_CATALOG).run(
#     model_name=bbp_brain_region_alone_model_description.name,
#     joint_bc=NexusBucketConfiguration(organisation="bbp", project="atlas", deployment=deployment),
#     bucket_list_rev=[
#         (NexusBucketConfiguration(
#             organisation="neurosciencegraph", project="datamodels",
#                                   deployment=deployment), None)
#     ],
#     target_type="BrainRegion"
# )
#
# print(e)

# Output Embedding Model data catalog id
# "https://bbp.epfl.ch/resources/bbp/atlas/_/c2f6d339-2231-4eee-b5fb-7e0112b4dc7d"

# ----------------------------------------------- 4 -----------------------------------------------

# AT STEP REGISTER EMBEDDINGS THIS IS WHERE THE TAG CREATION HAPPENS AND IT PROPAGATED AT FURTHER
# STEPS. IF TAG FORMAT SHOULD BE UPDATED, it's here

# embedding_tag, vector_dimension = ModelRegistrationPipeline.get_step(Step.REGISTER_EMBEDDINGS).run(
#     model_bc=NexusBucketConfiguration(
#         organisation="neurosciencegraph", project="datamodels", deployment=deployment
#     ),
#     model_description=bbp_brain_region_alone_model_description,
#     entity_type="BrainRegion"
# )

# ----------------------------------------------- 5 -----------------------------------------------

# embedding_tag = "27fb56d6-15ad-4df4-a82e-d52f34e19600?rev=1"
# vector_dimension = 32
#
# view_id = ModelRegistrationPipeline.get_step(Step.REGISTER_SIMILARITY_VIEW).run(
#     bucket_configuration=NexusBucketConfiguration(
#         organisation="neurosciencegraph", project="datamodels", deployment=deployment
#     ),
#     resource_tag=embedding_tag,
#     vector_dimension=vector_dimension
# )

# OUTPUT Similarity view id
# "https://bbp.epfl.ch/views/neurosciencegraph/datamodels/c1f07271-8c7b-4372-81b9-df7abce3bc8f"
