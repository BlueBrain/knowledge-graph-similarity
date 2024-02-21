from typing import Tuple, List, Optional

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration
from similarity_tools.data_classes.model_data import ModelData
from similarity_tools.registration.model_registration_step import ModelRegistrationStep
from similarity_tools.registration.step import Step

from similarity_tools.registration.steps.a_save_locally_model import step_1
from similarity_tools.registration.steps.b_register_model import step_2
from similarity_tools.registration.steps.c_register_embedding_model_catalog import step_3
from similarity_tools.registration.steps.d_register_embeddings import step_4
from similarity_tools.registration.steps.e_register_similarity_view import step_5
from similarity_tools.registration.steps.f_register_aggregated_similarity_view import step_6
from similarity_tools.registration.steps.g_register_non_boosted_stats import step_7
from similarity_tools.registration.steps.h_register_boosting_factors import step_8
from similarity_tools.registration.steps.i_register_boosting_view import step_9
from similarity_tools.registration.steps.j_register_aggregated_boosting_view import step_10
from similarity_tools.registration.steps.k_register_boosted_stats import step_11
from similarity_tools.registration.steps.l_register_stats_view import step_12


class ModelRegistrationPipeline:
    steps: List[ModelRegistrationStep] = [
        step_1, step_2, step_3,
        step_4, step_5, step_6,
        step_7, step_8, step_9,
        step_10, step_11, step_12
    ]

    @staticmethod
    def get_step(step: Step) -> ModelRegistrationStep:
        return ModelRegistrationPipeline.steps[step.value - 1]

    # @staticmethod
    # def run(
    #     model_description: ModelDescription,
    #     model_bc: NexusBucketConfiguration,
    #     embedding_bc: NexusBucketConfiguration,
    #     start_position: Step = Step.REGISTER_MODEL,
    #     data: ModelData = None
    # ):
    #
    #     if start_position == Step.SAVE_MODEL and data is None:
    #         raise Exception("Cannot save model without data to run it from")
    #
    #     if start_position == Step.SAVE_MODEL:
    #
    #         ModelRegistrationPipeline.get_step(Step.SAVE_MODEL).run(
    #             model_description=model_description,
    #             model_data=data,
    #         )
    #
    #     if start_position == Step.REGISTER_MODEL:
    #         model = ModelRegistrationPipeline.get_step(Step.REGISTER_MODEL).run(
    #             model_description=model_description,
    #             model_bc=model_bc
    #         )
    #     else:
    #         model = None
    #
    #     # TODO THERE SHOULD BE A REV UPDATE HERE
    #
    #     for step in ModelRegistrationPipeline.steps[start_position:]:
    #         step.run(
    #             model_description=model_description,
    #             model=model,
    #             model_bc=model_bc,
    #             embedding_bc=embedding_bc
    #         )
    #
    # @staticmethod
    # def run_many_models(
    #         model_descriptions: List[ModelDescription],
    #         model_bc: NexusBucketConfiguration,
    #         embedding_bc: NexusBucketConfiguration,
    #         start_position=Step.REGISTER_MODEL,
    #         data: Optional[ModelData] = None
    # ):
    #
    #     for model_description in model_descriptions:
    #         ModelRegistrationPipeline.run(
    #             model_description=model_description,
    #             model_bc=model_bc,
    #             embedding_bc=embedding_bc,
    #             start_position=start_position,
    #             data=data
    #         )
