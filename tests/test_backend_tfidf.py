"""Unit tests for the TF-IDF backend in Annif"""

import annif
import annif.backend


def test_tfidf_load_documents(
        datadir,
        document_corpus,
        project_with_vectorizer):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    tfidf.load_corpus(document_corpus, project_with_vectorizer)
    assert len(tfidf._index) > 0
    assert datadir.join('tfidf-index').exists()
    assert datadir.join('tfidf-index').size() > 0


def test_tfidf_analyze(datadir, project_with_vectorizer):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project_with_vectorizer)

    assert len(results) == 10
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]


def test_tfidf_analyze_unknown(datadir, project_with_vectorizer):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.analyze("abcdefghijk",
                            project_with_vectorizer)  # unknown word

    assert len(results) == 0
