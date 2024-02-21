from typing import Callable, List, Any, Tuple, Optional

from similarity_tools.helpers.logger import logger
from similarity_tools.registration.step import Step


class ModelRegistrationStep:
    step: Step

    function_call: Callable

    log_message: str

    def __init__(self, function_call: Callable, step: Step, log_message: str):
        self.step = step
        self.function_call = function_call
        self.log_message = log_message

    def run(self, **kwargs) -> Any:
        self.log(**kwargs)
        return self.function_call(**kwargs)

    def log(self, **kwargs):

        letter = chr(ord('@') + self.step.value)
        log_message = f"{letter}. {self.log_message}"

        model_description = kwargs.get("model_description", None)

        if model_description is not None:
            log_message += f" for model {model_description.name}"

        logger.info(log_message)
