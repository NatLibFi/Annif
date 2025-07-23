"""Support for exclude/include rules for subject vocabularies"""

from rdflib import RDF, Graph, URIRef
from rdflib.namespace import SKOS

from annif.exception import ConfigurationException


def uris_by_type(graph: Graph, type: str) -> list[str]:
    return [str(uri) for uri in graph.subjects(RDF.type, URIRef(type))]


def uris_by_scheme(graph: Graph, type: str) -> list[str]:
    return [str(uri) for uri in graph.subjects(SKOS.inScheme, URIRef(type))]


def uris_by_collection(graph: Graph, type: str) -> list[str]:
    return [str(uri) for uri in graph.objects(URIRef(type), SKOS.member)]


def kwargs_to_exclude_uris(graph: Graph, kwargs: dict[str, str]) -> set[str]:
    exclude_uris = set()
    for key, value in kwargs.items():
        vals = value.split("|")
        if key == "exclude":
            if "*" in vals:
                exclude_uris.update(uris_by_type(graph, SKOS.Concept))
            else:
                exclude_uris.update(vals)
        elif key == "exclude_type":
            for val in vals:
                exclude_uris.update(uris_by_type(graph, val))
        elif key == "exclude_scheme":
            for val in vals:
                exclude_uris.update(uris_by_scheme(graph, val))
        elif key == "exclude_collection":
            for val in vals:
                exclude_uris.update(uris_by_collection(graph, val))
        elif key == "include":
            for val in vals:
                exclude_uris.remove(val)
        elif key == "include_type":
            for val in vals:
                for uri in uris_by_type(graph, val):
                    exclude_uris.remove(uri)
        elif key == "include_scheme":
            for val in vals:
                for uri in uris_by_scheme(graph, val):
                    exclude_uris.remove(uri)
        elif key == "include_collection":
            for val in vals:
                for uri in uris_by_collection(graph, val):
                    exclude_uris.remove(uri)
        else:
            raise ConfigurationException(f"unknown vocab keyword argument {key}")
    return exclude_uris
