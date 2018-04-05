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
    return proj


def test_fasttext_load_subjects(datadir, subject_corpus, project):
    fasttext_type = annif.backend.get_backend_type("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        params={
            'limit': 50,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    fasttext.load_subjects(subject_corpus, project)
    assert fasttext._model is not None
    assert datadir.join('backends/fasttext/model.bin').exists()
    assert datadir.join('backends/fasttext/model.bin').size() > 0


def test_fasttext_analyze(datadir, project):
    fasttext_type = annif.backend.get_backend_type("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        params={
            'limit': 50,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    results = fasttext.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) == 50
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]
