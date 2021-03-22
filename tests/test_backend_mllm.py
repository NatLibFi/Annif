"""Unit tests for the MLLM backend in Annif"""

import annif
import annif.backend
from annif.backend.mllm import MLLMModel
import numpy as np


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


def test_mllm_default_params(project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={},
        project=project)

    expected_default_params = {
        'limit': 100,  # from AnnifBackend class
        'min_samples_leaf': 20,
        'max_leaf_nodes': 1000,
        'max_samples': 0.9
    }
    actual_params = mllm.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_mllm_train(datadir, document_corpus, project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    mllm.train(document_corpus)
    assert mllm._model is not None
    assert datadir.join('mllm-train.gz').exists()
    assert datadir.join('mllm-train.gz').size() > 0
    assert datadir.join('mllm-model.gz').exists()
    assert datadir.join('mllm-model.gz').size() > 0


def test_mllm_train_cached(datadir, project):
    modelfile = datadir.join('mllm-model.gz')
    assert modelfile.exists()

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    mllm.train("cached")
    assert mllm._model is not None
    assert modelfile.exists()
    assert modelfile.size() > 0
    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_mllm_suggest(project):
    mllm_type = annif.backend.get_backend('mllm')
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 8, 'language': 'fi'},
        project=project)

    results = mllm.suggest("""Arkeologia on tieteenala, jota sanotaan joskus
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0
    assert len(results) <= 8
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in hits]
    assert 'arkeologia' in [result.label for result in hits]


def test_mllm_suggest_no_matches(project):
    mllm_type = annif.backend.get_backend('mllm')
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 8, 'language': 'fi'},
        project=project)

    results = mllm.suggest("Nothing matches this.")

    assert len(results) == 0


def test_mllm_hyperopt(project, document_corpus):
    mllm_type = annif.backend.get_backend('mllm')
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    optimizer = mllm.get_hp_optimizer(document_corpus, metric='NDCG')
    optimizer.optimize(n_trials=3, n_jobs=1, results_file=None)
