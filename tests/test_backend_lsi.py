"""Unit tests for the LSI backend in Annif"""

import annif
import annif.backend


def test_lsi_load_documents(
        datadir,
        document_corpus,
        project_with_vectorizer):
    lsi_type = annif.backend.get_backend("lsi")
    lsi = lsi_type(
        backend_id='lsi',
        params={'limit': 10, 'num_topics': 100},
        datadir=str(datadir))

    lsi.load_corpus(document_corpus, project_with_vectorizer)
    assert len(lsi._index) > 0
    assert datadir.join('lsi-index').exists()
    assert datadir.join('lsi-index').size() > 0


def test_lsi_analyze(datadir, project_with_vectorizer):
    lsi_type = annif.backend.get_backend("lsi")
    lsi = lsi_type(
        backend_id='lsi',
        params={'limit': 10, 'num_topics': 100},
        datadir=str(datadir))

    results = lsi.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project_with_vectorizer)

    assert len(results) == 10
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]


def test_lsi_analyze_unknown(datadir, project_with_vectorizer):
    lsi_type = annif.backend.get_backend("lsi")
    lsi = lsi_type(
        backend_id='lsi',
        params={'limit': 10},
        datadir=str(datadir))

    results = lsi.analyze("abcdefghijk",
                          project_with_vectorizer)  # unknown word

    assert len(results) == 0
