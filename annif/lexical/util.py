"""Utility methods for lexical algorithms"""
from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from rdflib import URIRef
from rdflib.namespace import SKOS
from scipy.sparse import csc_matrix, lil_matrix

if TYPE_CHECKING:
    from rdflib.graph import Graph

    from annif.vocab import AnnifVocabulary


def get_subject_labels(
    graph: Graph, uri: str, properties: list[URIRef], language: str
) -> list[str]:
    return [
        str(label)
        for prop in properties
        for label in graph.objects(URIRef(uri), prop)
        if label.language == language
    ]


def make_relation_matrix(
    graph: Graph, vocab: AnnifVocabulary, property: URIRef
) -> csc_matrix:
    n_subj = len(vocab.subjects)
    matrix = lil_matrix((n_subj, n_subj), dtype=bool)

    for subj, obj in graph.subject_objects(property):
        subj_id = vocab.subjects.by_uri(str(subj), warnings=False)
        obj_id = vocab.subjects.by_uri(str(obj), warnings=False)
        if subj_id is not None and obj_id is not None:
            matrix[subj_id, obj_id] = True

    return csc_matrix(matrix)


def make_collection_matrix(graph: Graph, vocab: AnnifVocabulary) -> csc_matrix:
    # make an index with all collection members
    c_members = collections.defaultdict(list)
    for coll, member in graph.subject_objects(SKOS.member):
        member_id = vocab.subjects.by_uri(str(member), warnings=False)
        if member_id is not None:
            c_members[str(coll)].append(member_id)

    c_matrix = lil_matrix((len(c_members), len(vocab.subjects)), dtype=bool)

    # populate the matrix for collection -> subject_id
    for c_id, members in enumerate(c_members.values()):
        c_matrix[c_id, members] = True

    return csc_matrix(c_matrix)
