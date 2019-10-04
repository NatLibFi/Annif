"""Unit tests for the fastText backend in Annif"""

import pytest
import annif.backend
import annif.corpus
from annif.exception import NotSupportedException

fasttext = pytest.importorskip("annif.backend.fasttext")


def test_fasttext_default_params(datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        config_params={},
        datadir=str(datadir))

    expected_default_params = {
        'limit': 100,
        'chunksize': 1,
        'dim': 100,
        'lr': 0.25,
        'epoch': 5,
        'loss': 'hs',
    }
    actual_params = fasttext.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_fasttext_train(datadir, document_corpus, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        config_params={
            'limit': 50,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    fasttext.train(document_corpus, project)
    assert fasttext._model is not None
    assert datadir.join('fasttext-model').exists()
    assert datadir.join('fasttext-model').size() > 0


def test_fasttext_train_unknown_subject(tmpdir, datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        config_params={
            'limit': 50,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("nonexistent\thttp://example.com/nonexistent\n" +
                  "arkeologia\thttp://www.yso.fi/onto/yso/p1265")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    fasttext.train(document_corpus, project)
    assert fasttext._model is not None
    assert datadir.join('fasttext-model').exists()
    assert datadir.join('fasttext-model').size() > 0


def test_fasttext_train_nodocuments(tmpdir, datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        config_params={
            'limit': 50,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    empty_file = tmpdir.ensure('empty.tsv')
    empty_document_corpus = annif.corpus.DocumentFile(str(empty_file))

    with pytest.raises(NotSupportedException) as excinfo:
        fasttext.train(empty_document_corpus, project)
    assert 'training backend fasttext with no documents' in str(excinfo.value)


def test_fasttext_suggest(datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        config_params={
            'limit': 50,
            'chunksize': 1,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    results = fasttext.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]
