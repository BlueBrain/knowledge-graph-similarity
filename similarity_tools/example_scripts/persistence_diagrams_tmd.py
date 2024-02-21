from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

from similarity_tools.building.model_impl.tmd_model.tmd_model import TMDModel

from similarity_tools.building.model_data_impl.morphology_models import ModelDataMorphologyModels
from similarity_tools.building.model_data_impl.neuron_morphologies_query import \
    NeuronMorphologiesQuery

from similarity_tools.building.model_impl.tmd_model.tmd_model_with_mm import TMDModelWithMM, \
    UnscaledTMDModelWithMM

import numpy as np

re_download = True
re_compute = True

deployment = Deployment.PRODUCTION

# bucket_models = NexusBucketConfiguration(
#     organisation="bbp", project="mmb-point-neuron-framework-model", deployment=deployment
# )
#
# test = UnscaledTMDModelWithMM(
#     model_data=ModelDataMorphologyModels(bucket_configuration=bucket_models),
#     re_compute=re_compute, re_download=re_download
# )
# print(test.run())

bucket_morphologies = NexusBucketConfiguration(
    organisation="public", project="thalamus", deployment=deployment
)

NeuronMorphologiesQuery.LIMIT = 2

morphology_tmd = TMDModel(
    model_data=NeuronMorphologiesQuery(
        org=bucket_morphologies.organisation, project=bucket_morphologies.project,
        get_annotations=False
    ),
    re_compute=re_compute, re_download=re_download
)

# See building/model_impl/tmd_model/tmd_model.py
# uses Vectorisation.build_vectors_from_tmd_implementations
diagrams = morphology_tmd.run()
# 3 per morphology: persistence image data, betti curve, life entropy curve

for id_, (persistence_image_data, betti_curve, life_entropy_curve) in diagrams.items():
    print(np.shape(persistence_image_data), np.shape(betti_curve), np.shape(life_entropy_curve))
