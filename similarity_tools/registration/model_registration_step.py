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
