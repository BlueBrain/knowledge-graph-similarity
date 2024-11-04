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

from similarity_tools.helpers.bucket_configuration import NexusBucketConfiguration


new_tag = "v4"
test = NexusBucketConfiguration("bbp", "inference-rules", True).allocate_forge_session()

a = "https://bbp.epfl.ch/neurosciencegraph/data/5d04995a-6220-4e82-b847-8c3a87030e0b"  # hierarchy
b = "https://bbp.epfl.ch/neurosciencegraph/data/abb1949e-dc16-4719-b43b-ff88dabc4cb8"  # neuron m
c = "https://bbp.epfl.ch/neurosciencegraph/data/9d64dc0d-07d1-4624-b409-cdc47ccda212"  # gen sim br
to_tag = [test.retrieve(el) for el in [a, b, c]]
test.tag(to_tag, new_tag)
