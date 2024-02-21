from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration

from similarity_tools.data_classes.model_data import ModelData
from similarity_tools.helpers.logger import logger


class ModelDataMorphologyModels(ModelData):

    def __init__(
            self, bucket_configuration: NexusBucketConfiguration,
            src_data_dir=None, dst_data_dir=None,
    ):
        super().__init__(
            src_data_dir=src_data_dir, dst_data_dir=dst_data_dir,
            deployment=bucket_configuration.deployment,
            org=bucket_configuration.organisation, project=bucket_configuration.project
        )

        logger.info(">  Load morphological models from Nexus")

        self.forge = bucket_configuration.allocate_forge_session()

        self.data = self.forge.search({"type": "CanonicalMorphologyModel"}, limit=1500)

