from typing import Optional, Dict, List, Callable

from kgforge.core import KnowledgeGraphForge, Resource
import jwt
from kgforge.specializations.resources import Dataset

from similarity_tools.helpers.logger import logger
from similarity_tools.registration.registration_exception import SimilarityToolsException


def _fetch_one_dict(
    forge: KnowledgeGraphForge, type_str: str, search_dict: Dict, entity_name: Optional[str] = None
):
    result = forge.search(search_dict)

    if result is None:
        raise SimilarityToolsException(f"Could not query for existing {type_str}s")
    if len(result) == 0:
        logger.info(f"{type_str.capitalize()} {entity_name if entity_name else ''} does not exist")
        return None
    if len(result) != 1:
        logger.warning(f"More than one {type_str} match, returning first one")

    return result[0]


def _fetch_one(
        entity_name: str, entity_type: str,
        forge: KnowledgeGraphForge, type_str: str, entity_rev: Optional[int] = None
) -> Optional[Resource]:

    search_dict = {
        "name": entity_name,
        "type": entity_type,
        "_deprecated": False
    }

    if entity_rev is not None:
        search_dict["_rev"] = entity_rev

    return _fetch_one_dict(forge, type_str, search_dict)


def _get_agent(forge) -> Resource:
    """Create a Nexus agent."""
    token = forge._store.token
    agent_data = jwt.decode(token, options={"verify_signature": False})
    agent = forge.reshape(
        forge.from_json(agent_data),
        keep=["name", "email", "sub", "preferred_username"]
    )
    agent.id = f"{forge._store.endpoint}/realms/bbp/users/{agent.preferred_username}"
    agent.type = "Person"
    return agent


def add_contribution(resource: Dataset, forge: KnowledgeGraphForge) -> Dataset:
    resource.add_contribution(_get_agent(forge), versioned=False)
    role = forge.from_json({
            "id": "http://purl.obolibrary.org/obo/CRO_0000064",
            "label": "software engineering role"
    })
    resource.contribution.hadRole = role
    return resource


def _persist(
        entities: List[Resource], creation: bool, schema_id: str,
        forge: KnowledgeGraphForge, tag: Optional[str], obj_str: str
):
    verb_a, verb_b, verb_c = \
        ("creating", "created", "create") if creation else ("updating", "updated", "update")

    if len(entities) > 0:
        logger.info(f">  {verb_a.capitalize()} {obj_str}: {len(entities)}")
        fc: Callable[[List[Resource]], None] = forge.register if creation else forge.update

        fc(entities, schema_id=schema_id)
        success = all([e._last_action.succeeded for e in entities])

        if success:

            logger.info(f">  {verb_b.capitalize()} {obj_str}: {len(entities)}")
            logger.info(f">  Tagging {verb_b} {obj_str}...")

            def synchronize(res: Resource):
                res._synchronized = True
                return res

            if not creation:
                entities = list(map(synchronize, entities))

            if tag:
                forge.tag(entities, tag)
        else:
            errors = [
                e for e in map(lambda res: res._last_action.error, entities) if e is not None
            ]
            for error in errors:
                logger.error(error)

    else:
        logger.info(f"> No {obj_str} to {verb_c}")
