"""Unit tests for the MLLM backend in Annif"""

import annif
import annif.backend
from annif.backend.mllm import TokenSet, TokenSetIndex, MLLMModel


def test_mllm_tokenset():
    tokens = [1, 3, 5]
    tset = TokenSet(tokens)
    assert tset.subject_id is None
    assert not tset.is_pref
    assert len(tset) == len(tokens)
    assert sorted(list(tset)) == sorted(tokens)
    assert tset.contains(TokenSet(tokens))
    assert tset.contains(TokenSet([1]))
    assert not tset.contains(TokenSet([0]))
    assert tset.sample() in tokens


def test_mllm_tokenset_empty_sample():
    assert TokenSet([]).sample() is None


def test_mllm_tokensetindex():
    index = TokenSetIndex()
    assert len(index) == 0
    tset13 = TokenSet([1, 3], subject_id=1)
    index.add(tset13)
    assert len(index) == 1
    index.add(TokenSet([]))  # add empty
    assert len(index) == 1
    tset2 = TokenSet([2])
    index.add(tset2)
    tset23 = TokenSet([2, 3], subject_id=2)
    index.add(tset23)
    tset3 = TokenSet([3], subject_id=3, is_pref=True)
    index.add(tset3)
    tset34 = TokenSet([3, 4], subject_id=3, is_pref=False)
    index.add(tset34)
    tset5 = TokenSet([5])
    index.add(tset5)

    result = index.search(TokenSet([1, 2, 3, 4]))
    assert len(result) == 4

    assert (tset13, 0) in result
    assert (tset2, 1) in result
    assert (tset23, 0) in result
    assert (tset3, 2) in result

    assert tset34 not in [r[0] for r in result]
    assert tset5 not in [r[0] for r in result]


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
    matrix = model._related_matrix
    assert matrix.shape == (130, 130)  # 130x130 subjects
    assert matrix.sum() == 112  # 112 skos:related triples


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
