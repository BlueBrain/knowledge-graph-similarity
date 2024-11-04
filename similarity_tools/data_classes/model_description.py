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

class ModelDescription:
    def __init__(self, model_dictionary):
        self.name = model_dictionary["name"]
        self.description = model_dictionary["description"]
        self.filename = model_dictionary["filename"]
        self.label = model_dictionary["label"]
        self.distance = model_dictionary["distance"]
        self.model = model_dictionary["model"]
        self.model_rev: int = model_dictionary["rev"]
