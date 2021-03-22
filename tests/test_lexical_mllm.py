"""Unit tests for the MLLM model in Annif"""

import numpy as np
from annif.lexical.mllm import MLLMModel


def test_mllmmodel_prepare_terms(vocabulary):
    model = MLLMModel()
    graph = vocabulary.as_graph()
    params = {'language': 'fi', 'use_hidden_labels': True}
    terms, subject_ids = model._prepare_terms(
        graph, vocabulary, params)
    assert len(terms) == 164  # 130 prefLabels + 34 altLabels
    assert len(subject_ids) == 130  # 130 subjects


def test_mllmmodel_prepare_relations(vocabulary):
    model = MLLMModel()
    graph = vocabulary.as_graph()
    model._prepare_relations(graph, vocabulary)

    b_matrix = model._broader_matrix.todense()
    assert b_matrix.shape == (130, 130)  # 130x130 subjects
    assert b_matrix.sum() == 51  # 51 skos:broader triples

    n_matrix = model._narrower_matrix.todense()
    assert n_matrix.shape == (130, 130)  # 130x130 subjects
    assert n_matrix.sum() == 51  # 51 skos:narrower triples

    # broader is inverse of narrower, check by transposing!
    assert np.array_equal(n_matrix.T, b_matrix)

    r_matrix = model._related_matrix.todense()
    assert r_matrix.shape == (130, 130)  # 130x130 subjects
    assert r_matrix.sum() == 112  # 112 skos:related triples

    # related is symmetric, check by transposing!
    assert np.array_equal(r_matrix.T, r_matrix)
