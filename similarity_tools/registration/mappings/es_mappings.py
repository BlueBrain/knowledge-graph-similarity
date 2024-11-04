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

import copy

SIMILARITY_VIEW_MAPPING = {
    "properties": {
        "@id": {
            "type": "keyword"
        },
        "@type": {
            "type": "keyword"
        },
        "derivation": {
            "properties": {
                "entity": {
                    "properties": {
                        "@id": {
                            "type": "keyword"
                        },
                        "@type": {
                            "type": "keyword"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        },
        "embedding": {
            "type": "dense_vector"
        },
        "generation": {
            "properties": {
                "activity": {
                    "properties": {
                        "used": {
                            "properties": {
                                "@id": {
                                    "type": "keyword"
                                },
                                "@type": {
                                    "type": "keyword"
                                }
                            },
                            "type": "nested"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        }
    }
}

SIMILARITY_VIEW_BINARY_MAPPING = {
    "properties": {
        "@id": {
            "type": "keyword"
        },
        "@type": {
            "type": "keyword"
        },
        "derivation": {
            "properties": {
                "entity": {
                    "properties": {
                        "@id": {
                            "type": "keyword"
                        },
                        "@type": {
                            "type": "keyword"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        },
        "embedding": {
            "doc_values": True,
            "store": True,
            "type": "binary"
        },
        "generation": {
            "properties": {
                "activity": {
                    "properties": {
                        "used": {
                            "properties": {
                                "@id": {
                                    "type": "keyword"
                                },
                                "@type": {
                                    "type": "keyword"
                                }
                            },
                            "type": "nested"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        }
    }
  }

BOOSTING_VIEW_MAPPING = {
    "properties": {
        "@id": {
            "type": "keyword"
        },
        "@type": {
            "type": "keyword"
        },
        "value": {
            "type": "float"
        },
        "scriptScore": {
            "type": "keyword"
        },
        "vectorParameter": {
            "type": "keyword"
        },
        "derivation": {
            "properties": {
                "entity": {
                    "properties": {
                        "@id": {
                            "type": "keyword"
                        },
                        "_rev": {
                            "type": "long"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        },
        "generation": {
            "properties": {
                "activity": {
                    "properties": {
                        "used": {
                            "properties": {
                                "@id": {
                                    "type": "keyword"
                                },
                                "_rev": {
                                    "type": "long"
                                }
                            },
                            "type": "nested"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        }
    }
}

STATS_VIEW_MAPPING = {
    "properties": {
        "@id": {
            "type": "keyword"
        },
        "@type": {
            "type": "keyword"
        },
        "boosted": {
            "type": "boolean"
        },
        "scriptScore": {
            "type": "keyword"
        },
        "vectorParameter": {
            "type": "keyword"
        },
        "derivation": {
            "properties": {
                "entity": {
                    "properties": {
                        "@id": {
                            "type": "keyword"
                        }
                    },
                    "type": "nested"
                }
            },
            "type": "nested"
        },
        "series": {
            "properties": {
                "statistic": {
                    "type": "keyword"
                },
                "value": {
                    "type": "float"
                }
            },
            "type": "nested"
        }
    }
}


def get_es_view_mappings(dimension):
    mapping = copy.deepcopy(SIMILARITY_VIEW_MAPPING)
    mapping["properties"]["embedding"]["dims"] = dimension
    return mapping


def get_es_view_binary_mappings():
    mapping = copy.deepcopy(SIMILARITY_VIEW_BINARY_MAPPING)
    return mapping
