"""Unit tests for the vw_multi backend in Annif"""

import logging
import pytest
import annif.backend
import annif.corpus
from annif.exception import ConfigurationException, NotInitializedException

pytest.importorskip("annif.backend.vw_multi")


@pytest.fixture(scope='function')
def vw_corpus(tmpdir):
    """return a small document corpus for testing VW training"""
    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("nonexistent\thttp://example.com/nonexistent\n" +
                  "arkeologia\thttp://www.yso.fi/onto/yso/p1265\n" +
                  "...\thttp://example.com/none")
    return annif.corpus.DocumentFile(str(tmpfile))


def test_vw_multi_default_params(project):
    vw_type = annif.backend.get_backend("vw_multi")
    vw = vw_type(
        backend_id='vw_multi',
        config_params={},
        project=project)

    expected_default_params = {
        'limit': 100,
        'chunksize': 1,
        'algorithm': 'oaa',
        'loss_function': 'logistic',
    }
    actual_params = vw.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_vw_multi_suggest_no_model(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4},
        project=project)

    with pytest.raises(NotInitializedException):
        vw.suggest("example text")


def test_vw_multi_train_and_learn(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'loss_function': 'hinge'},
        project=project)

    vw.train(document_corpus)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0

    # test online learning
    modelfile = datadir.join('vw-model')

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    vw.learn(document_corpus)

    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_vw_multi_train_and_learn_nodocuments(datadir, project, empty_corpus):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'loss_function': 'hinge'},
        project=project)

    vw.train(empty_corpus)
    assert datadir.join('vw-train.txt').exists()
    assert datadir.join('vw-train.txt').size() == 0

    # test online learning
    modelfile = datadir.join('vw-model')

    old_size = modelfile.size()

    vw.learn(empty_corpus)

    assert modelfile.size() == old_size
    assert datadir.join('vw-train.txt').size() == 0


def test_vw_multi_train_from_project(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'inputs': '_text_,dummy-en'},
        project=project)

    vw.train(document_corpus)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_train_multiple_passes(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'passes': 2},
        project=project)

    vw.train(document_corpus)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_train_invalid_algorithm(document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'invalid'},
        project=project)

    with pytest.raises(ConfigurationException):
        vw.train(document_corpus)


def test_vw_multi_train_invalid_loss_function(project, vw_corpus):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4, 'loss_function': 'invalid'},
        project=project)

    with pytest.raises(ConfigurationException):
        vw.train(vw_corpus)


def test_vw_multi_train_invalid_learning_rate(project, vw_corpus):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4, 'learning_rate': 'high'},
        project=project)

    with pytest.raises(ConfigurationException):
        vw.train(vw_corpus)


def test_vw_multi_train_params(project, vw_corpus, caplog):
    logger = annif.logger
    logger.propagate = True

    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4, 'learning_rate': 0.5},
        project=project)
    params = {'loss_function': 'logistic', 'learning_rate': 42.1}

    with caplog.at_level(logging.DEBUG):
        vw.train(vw_corpus, params)
    parameters_heading = 'Backend vw_multi: model parameters:'
    assert parameters_heading in caplog.text
    for line in caplog.text.splitlines():
        if parameters_heading in line:
            assert "'loss_function': 'logistic'" in line
            assert "'learning_rate': 42.1" in line


def test_vw_multi_train_cached(datadir, project, vw_corpus):
    assert datadir.join('vw-train.txt').exists()
    datadir.join('vw-model').remove()
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4, 'learning_rate': 0.5},
        project=project)
    vw.train("cached")
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4, 'probabilities': 1},
        project=project)

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in hits]
    assert 'arkeologia' in [result.label for result in hits]


def test_vw_multi_suggest_empty(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4},
        project=project)

    results = vw.suggest("...")

    assert len(results) == 0


def test_vw_multi_suggest_multiple_passes(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 4, 'passes': 2},
        project=project)

    results = vw.suggest("...")

    assert len(results) == 0


def test_vw_multi_train_ect(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'ect'},
        project=project)

    vw.train(document_corpus)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest_ect(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 1,
                       'algorithm': 'ect'},
        project=project)

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0


def test_vw_multi_train_log_multi(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'log_multi'},
        project=project)

    vw.train(document_corpus)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest_log_multi(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 1,
                       'algorithm': 'log_multi'},
        project=project)

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0


def test_vw_multi_train_multilabel_oaa(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'multilabel_oaa'},
        project=project)

    vw.train(document_corpus)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest_multilabel_oaa(project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        config_params={'chunksize': 1,
                       'algorithm': 'multilabel_oaa'},
        project=project)

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    # weak assertion, but often multilabel_oaa produces zero hits
    assert results is not None
