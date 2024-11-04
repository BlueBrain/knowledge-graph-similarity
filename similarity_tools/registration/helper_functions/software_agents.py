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

from bluegraph import version as bg_version
import sys
import subprocess


def get_git_revision_hash(short=True) -> str:
    arr = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
    return subprocess.check_output(arr).decode('ascii').strip()


def get_git_branch():
    output = subprocess.check_output(['git', 'branch']).decode('ascii').strip()
    branch = next(a for a in output.split('\n') if "*" in a)
    return branch[branch.find('*') + 2:]


def _software_agent_bluegraph():
    return {
        "type": "SoftwareAgent",
        "description": "Unifying Python framework for graph analytics and co-occurrence analysis.",
        "name": "BlueGraph",
        "softwareSourceCode": {
            "type": "SoftwareSourceCode",
            "codeRepository": "https://github.com/BlueBrain/BlueGraph",
            "programmingLanguage": "Python",
            "runtimePlatform": get_python_version(),
            "version": get_bluegraph_version()
        }
    }


def _software_agent_similarity_tools():
    return {
        "type": "SoftwareAgent",
        "description": "Tools for performing registration of data for similarity-based inference.",
        "name": "KG Inference Similarity",
        "softwareSourceCode": {
            "type": "SoftwareSourceCode",
            "codeRepository": "https://bbpgitlab.epfl.ch/dke/apps/kg-inference-similarity",
            "branch": get_git_branch(),
            "commit": get_git_revision_hash(),
            "programmingLanguage": "Python",
            "runtimePlatform": get_python_version()
        }
    }


def get_wasAssociatedWith(bluegraph: bool):
    return [_software_agent_similarity_tools()] if not bluegraph \
        else [_software_agent_similarity_tools(), _software_agent_bluegraph()]


def get_bluegraph_version():
    return bg_version.__version__


def get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"
