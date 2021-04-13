"""Utility methods for lexical algorithms"""

import collections
from rdflib import URIRef
from rdflib.namespace import SKOS
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix


def get_subject_labels(graph, uri, properties, language):
    return [str(label)
            for prop in properties
            for label in graph.objects(URIRef(uri), prop)
            if label.language == language]


def make_relation_matrix(graph, vocab, property):
    n_subj = len(vocab.subjects)
    matrix = lil_matrix((n_subj, n_subj), dtype=np.bool)

    for subj, obj in graph.subject_objects(property):
        subj_id = vocab.subjects.by_uri(str(subj), warnings=False)
        obj_id = vocab.subjects.by_uri(str(obj), warnings=False)
        if subj_id is not None and obj_id is not None:
            matrix[subj_id, obj_id] = True

    return csc_matrix(matrix)


def make_collection_matrix(graph, vocab):
    # make an index with all collection members
    c_members = collections.defaultdict(list)
    for coll, member in graph.subject_objects(SKOS.member):
        member_id = vocab.subjects.by_uri(str(member), warnings=False)
        if member_id is not None:
            c_members[str(coll)].append(member_id)

    c_matrix = lil_matrix((len(c_members), len(vocab.subjects)),
                          dtype=np.bool)

    # populate the matrix for collection -> subject_id
    for c_id, members in enumerate(c_members.values()):
        c_matrix[c_id, members] = True

    return csc_matrix(c_matrix)
