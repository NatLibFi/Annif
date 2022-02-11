"""Unit tests for the MLLM backend in Annif"""

import pytest
import annif
import annif.backend
from annif.exception import NotInitializedException
from annif.exception import NotSupportedException


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


def test_mllm_train(datadir, fulltext_corpus, project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    mllm.train(fulltext_corpus)
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


def test_mllm_train_nodocuments(project, empty_corpus):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    with pytest.raises(NotSupportedException) as excinfo:
        mllm.train(empty_corpus)
    assert 'training backend mllm with no documents' in str(excinfo.value)


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


def test_mllm_hyperopt(project, fulltext_corpus):
    mllm_type = annif.backend.get_backend('mllm')
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    optimizer = mllm.get_hp_optimizer(fulltext_corpus, metric='NDCG')
    optimizer.optimize(n_trials=3, n_jobs=1, results_file=None)


def test_mllm_train_cached_no_data(datadir, project):
    modelfile = datadir.join('mllm-model.gz')
    assert modelfile.exists()
    trainfile = datadir.join('mllm-train.gz')
    trainfile.remove()

    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 10, 'language': 'fi'},
        project=project)

    with pytest.raises(NotInitializedException):
        mllm.train("cached")


def test_mllm_suggest_no_model(datadir, project):
    mllm_type = annif.backend.get_backend('mllm')
    mllm = mllm_type(
        backend_id='mllm',
        config_params={'limit': 8, 'language': 'fi'},
        project=project)

    datadir.join('mllm-model.gz').remove()
    with pytest.raises(NotInitializedException):
        mllm.suggest("example text")
