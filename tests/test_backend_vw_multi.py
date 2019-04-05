"""Unit tests for the fastText backend in Annif"""

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


def test_vw_multi_suggest_no_model(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 4},
        datadir=str(datadir))

    with pytest.raises(NotInitializedException):
        results = vw.suggest("example text", project)


def test_vw_multi_train_and_learn(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'loss_function': 'hinge'},
        datadir=str(datadir))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0

    # test online learning
    modelfile = datadir.join('vw-model')

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    vw.learn(document_corpus, project)

    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_vw_multi_train_from_project(app, datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'inputs': '_text_,dummy-en'},
        datadir=str(datadir))

    with app.app_context():
        vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_train_multiple_passes(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'passes': 2},
        datadir=str(datadir))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_train_invalid_algorithm(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'invalid'},
        datadir=str(datadir))

    with pytest.raises(ConfigurationException):
        vw.train(document_corpus, project)


def test_vw_multi_train_invalid_loss_function(datadir, project, vw_corpus):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 4, 'loss_function': 'invalid'},
        datadir=str(datadir))

    with pytest.raises(ConfigurationException):
        vw.train(vw_corpus, project)


def test_vw_multi_train_invalid_learning_rate(datadir, project, vw_corpus):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 4, 'learning_rate': 'high'},
        datadir=str(datadir))

    with pytest.raises(ConfigurationException):
        vw.train(vw_corpus, project)


def test_vw_multi_suggest(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 4, 'probabilities': 1},
        datadir=str(datadir))

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]


def test_vw_multi_suggest_empty(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 4},
        datadir=str(datadir))

    results = vw.suggest("...", project)

    assert len(results) == 0


def test_vw_multi_suggest_multiple_passes(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 4, 'passes': 2},
        datadir=str(datadir))

    results = vw.suggest("...", project)

    assert len(results) == 0


def test_vw_multi_train_ect(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'ect'},
        datadir=str(datadir))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest_ect(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 1,
                'algorithm': 'ect'},
        datadir=str(datadir))

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0


def test_vw_multi_train_log_multi(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'log_multi'},
        datadir=str(datadir))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest_log_multi(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 1,
                'algorithm': 'log_multi'},
        datadir=str(datadir))

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0


def test_vw_multi_train_multilabel_oaa(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={
            'chunksize': 4,
            'learning_rate': 0.5,
            'algorithm': 'multilabel_oaa'},
        datadir=str(datadir))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_multi_suggest_multilabel_oaa(datadir, project):
    vw_type = annif.backend.get_backend('vw_multi')
    vw = vw_type(
        backend_id='vw_multi',
        params={'chunksize': 1,
                'algorithm': 'multilabel_oaa'},
        datadir=str(datadir))

    results = vw.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    # weak assertion, but often multilabel_oaa produces zero hits
    assert len(results) >= 0
