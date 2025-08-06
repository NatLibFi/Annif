"""Support for exclude/include rules for subject vocabularies"""

from rdflib import RDF, Graph, URIRef
from rdflib.namespace import SKOS

import annif
from annif.exception import ConfigurationException

from .vocab import AnnifVocabulary

logger = annif.logger


def resolve_uri_or_curie(graph: Graph, value: str) -> URIRef:
    try:
        # Try to expand as CURIE using the graph's namespace manager
        return graph.namespace_manager.expand_curie(value)
    except ValueError:
        # Not a CURIE or prefix not defined; treat as full URI
        return URIRef(value)


def uris_by_type(graph: Graph, type_: str, action: str) -> list[str]:
    type_uri = resolve_uri_or_curie(graph, type_)
    uris = [str(uri) for uri in graph.subjects(RDF.type, type_uri)]
    if not uris:
        logger.warning(f"{action}: no concepts found with type {type_uri}")
    return uris


def uris_by_scheme(graph: Graph, scheme: str, action: str) -> list[str]:
    scheme_uri = resolve_uri_or_curie(graph, scheme)
    uris = [str(uri) for uri in graph.subjects(SKOS.inScheme, scheme_uri)]
    if not uris:
        logger.warning(f"{action}: no concepts found in scheme {scheme_uri}")
    return uris


def uris_by_collection(graph: Graph, collection: str, action: str) -> list[str]:
    collection_uri = resolve_uri_or_curie(graph, collection)
    uris = [str(uri) for uri in graph.objects(collection_uri, SKOS.member)]
    if not uris:
        logger.warning(f"{action}: no concepts found in collection {collection_uri}")
    return uris


def add_uris(
    graph: Graph, uris_func: callable, uris_set: set[str], vals: list[str], action: str
) -> None:
    for val in vals:
        uris_set.update(uris_func(graph, val, action))


def remove_uris(
    graph: Graph, uris_func: callable, uris_set: set[str], vals: list[str], action: str
) -> None:
    for val in vals:
        for uri in uris_func(graph, val, action):
            uris_set.discard(uri)


def kwargs_to_exclude_uris(vocab: AnnifVocabulary, kwargs: dict[str, str]) -> set[str]:
    exclude_uris = set()
    actions = {
        "exclude": lambda vals: exclude_uris.update(
            vals
            if "*" not in vals
            else uris_by_type(vocab.as_graph(), "skos:Concept", "exclude")
        ),
        "exclude_type": lambda vals: add_uris(
            vocab.as_graph(), uris_by_type, exclude_uris, vals, "exclude_type"
        ),
        "exclude_scheme": lambda vals: add_uris(
            vocab.as_graph(), uris_by_scheme, exclude_uris, vals, "exclude_scheme"
        ),
        "exclude_collection": lambda vals: add_uris(
            vocab.as_graph(),
            uris_by_collection,
            exclude_uris,
            vals,
            "exclude_collection",
        ),
        "include": lambda vals: exclude_uris.difference_update(vals),
        "include_type": lambda vals: remove_uris(
            vocab.as_graph(), uris_by_type, exclude_uris, vals, "include_type"
        ),
        "include_scheme": lambda vals: remove_uris(
            vocab.as_graph(), uris_by_scheme, exclude_uris, vals, "include_scheme"
        ),
        "include_collection": lambda vals: remove_uris(
            vocab.as_graph(),
            uris_by_collection,
            exclude_uris,
            vals,
            "include_collection",
        ),
    }

    for key, value in kwargs.items():
        vals = value.split("|")
        if key in actions:
            actions[key](vals)
        else:
            raise ConfigurationException(f"unknown vocab keyword argument {key}")

    return exclude_uris
