from similarity_tools.building.model_data_impl.morphology_models import ModelDataMorphologyModels

from similarity_tools.building.model_impl.tmd_model.tmd_model_with_mm import TMDModelWithMM, \
    UnscaledTMDModelWithMM

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration, Deployment

re_download = False
re_compute = True

deployment = Deployment.PRODUCTION

bucket_models = NexusBucketConfiguration(
    organisation="bbp", project="mmb-point-neuron-framework-model", deployment=deployment
)

test = UnscaledTMDModelWithMM(
    model_data=ModelDataMorphologyModels(bucket_configuration=bucket_models),
    re_compute=re_compute, re_download=re_download
)
print(test.run())
