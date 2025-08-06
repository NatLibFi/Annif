"""Support for exclude/include rules for subject vocabularies"""

from rdflib import RDF, Graph, URIRef
from rdflib.namespace import SKOS

from annif.exception import ConfigurationException

from .vocab import AnnifVocabulary


def uris_by_type(graph: Graph, type: str) -> list[str]:
    return [str(uri) for uri in graph.subjects(RDF.type, URIRef(type))]


def uris_by_scheme(graph: Graph, type: str) -> list[str]:
    return [str(uri) for uri in graph.subjects(SKOS.inScheme, URIRef(type))]


def uris_by_collection(graph: Graph, type: str) -> list[str]:
    return [str(uri) for uri in graph.objects(URIRef(type), SKOS.member)]


def add_uris(
    graph: Graph, uris_func: callable, uris_set: set[str], vals: list[str]
) -> None:
    for val in vals:
        uris_set.update(uris_func(graph, val))


def remove_uris(
    graph: Graph, uris_func: callable, uris_set: set[str], vals: list[str]
) -> None:
    for val in vals:
        for uri in uris_func(graph, val):
            uris_set.discard(uri)


def kwargs_to_exclude_uris(vocab: AnnifVocabulary, kwargs: dict[str, str]) -> set[str]:
    exclude_uris = set()
    actions = {
        "exclude": lambda vals: exclude_uris.update(
            vals if "*" not in vals else uris_by_type(vocab.as_graph(), SKOS.Concept)
        ),
        "exclude_type": lambda vals: add_uris(
            vocab.as_graph(), uris_by_type, exclude_uris, vals
        ),
        "exclude_scheme": lambda vals: add_uris(
            vocab.as_graph(), uris_by_scheme, exclude_uris, vals
        ),
        "exclude_collection": lambda vals: add_uris(
            vocab.as_graph(), uris_by_collection, exclude_uris, vals
        ),
        "include": lambda vals: exclude_uris.difference_update(vals),
        "include_type": lambda vals: remove_uris(
            vocab.as_graph(), uris_by_type, exclude_uris, vals
        ),
        "include_scheme": lambda vals: remove_uris(
            vocab.as_graph(), uris_by_scheme, exclude_uris, vals
        ),
        "include_collection": lambda vals: remove_uris(
            vocab.as_graph(), uris_by_collection, exclude_uris, vals
        ),
    }

    for key, value in kwargs.items():
        vals = value.split("|")
        if key in actions:
            actions[key](vals)
        else:
            raise ConfigurationException(f"unknown vocab keyword argument {key}")

    return exclude_uris
