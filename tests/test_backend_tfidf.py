"""Unit tests for the TF-IDF backend in Annif"""

import annif
import annif.backend
import annif.corpus
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest
import unittest.mock


@pytest.fixture(scope='module')
def datadir(tmpdir_factory):
    return tmpdir_factory.mktemp('data')


@pytest.fixture(scope='module')
def subject_corpus():
    subjdir = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects')
    return annif.corpus.SubjectDirectory(subjdir)


@pytest.fixture(scope='module')
def project(subject_corpus):
    proj = unittest.mock.Mock()
    proj.analyzer = annif.analyzer.get_analyzer('snowball(finnish)')
    proj.subjects = annif.corpus.SubjectIndex(subject_corpus)
    proj.vectorizer = TfidfVectorizer(tokenizer=proj.analyzer.tokenize_words)
    proj.vectorizer.fit([subj.text for subj in subject_corpus])
    return proj


def test_tfidf_load_subjects(datadir, subject_corpus, project):
    tfidf_type = annif.backend.get_backend_type("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    tfidf.load_subjects(subject_corpus, project)
    assert len(tfidf._index) > 0
    assert datadir.join('backends/tfidf/index').exists()
    assert datadir.join('backends/tfidf/index').size() > 0


def test_tfidf_analyze(datadir, project):
    tfidf_type = annif.backend.get_backend_type("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) == 10
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]


def test_tfidf_analyze_unknown(datadir, project):
    tfidf_type = annif.backend.get_backend_type("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.analyze("abcdefghijk", project)  # unknown word

    assert len(results) == 0
