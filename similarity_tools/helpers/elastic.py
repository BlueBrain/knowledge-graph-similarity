import json
from typing import Dict, Optional, List

from kgforge.core import KnowledgeGraphForge, Resource


class ElasticSearch:
    NO_LIMIT = 10000

    @staticmethod
    def get_all_documents_query():
        return {
            "size": ElasticSearch.NO_LIMIT,
            "query": {
                "term": {
                    "_deprecated": False
                }
            }
        }

    @staticmethod
    def get_all_documents(forge: KnowledgeGraphForge) -> Optional[List[Resource]]:
        """
        Retrieves all Resources that are indexed by the current elastic view endpoint of the forge
        instance
        @param forge: the forge instance
        @type forge: KnowledgeGraphForge
        @return:
        @rtype:  Optional[List[Resource]]
        """
        return forge.elastic(json.dumps(ElasticSearch.get_all_documents_query()))

    @staticmethod
    def get_by_ids(ids: List[str], forge: KnowledgeGraphForge) -> Optional[List[Resource]]:
        """

        @param ids: the list of ids of the resources to retrieve
        @type ids: List[str]
        @param forge: a forge instance
        @type forge: KnowledgeGraphForge
        @return: the list of Resources retrieved, if successful else None
        @rtype: Optional[List[Resource]]
        """
        q = {
            "size": ElasticSearch.NO_LIMIT,
            'query': {
                'bool': {
                    'filter': [
                        {'terms': {'@id': ids}}
                    ],
                    'must': [
                        {'match': {'_deprecated': False}}
                    ]
                }
            }
        }

        return forge.elastic(json.dumps(q), debug=False)