"""Unit tests for the TF-IDF backend in Annif"""

import annif
import annif.backend
import annif.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest
import unittest.mock


@pytest.fixture(scope='module')
def project(document_corpus):
    proj = unittest.mock.Mock()
    proj.analyzer = annif.analyzer.get_analyzer('snowball(finnish)')
    proj.subjects = annif.corpus.SubjectIndex(document_corpus)
    proj.vectorizer = TfidfVectorizer(tokenizer=proj.analyzer.tokenize_words)
    proj.vectorizer.fit([subj.text for subj in document_corpus.subjects])
    return proj


def test_tfidf_train(datadir, document_corpus, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    tfidf.train(document_corpus, project)
    assert len(tfidf._index) > 0
    assert datadir.join('tfidf-index').exists()
    assert datadir.join('tfidf-index').size() > 0


def test_tfidf_suggest(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) == 10
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]


def test_tfidf_suggest_unknown(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.suggest("abcdefghijk", project)  # unknown word

    assert len(results) == 0
