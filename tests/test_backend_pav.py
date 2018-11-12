"""Unit tests for the PAV backend in Annif"""

import annif.backend


def test_pav_load_documents(app, datadir, document_corpus, project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 1, 'sources': 'tfidf-fi,fasttext-fi'},
        datadir=str(datadir))

    with app.app_context():
        pav.load_corpus(document_corpus, project)
    assert datadir.join('pav-model-tfidf-fi').exists()
    assert datadir.join('pav-model-tfidf-fi').size() > 0
    assert datadir.join('pav-model-fasttext-fi').exists()
    assert datadir.join('pav-model-fasttext-fi').size() > 0


def test_pav_analyze(app, datadir, project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 1, 'sources': 'tfidf-fi,fasttext-fi'},
        datadir=str(datadir))

    results = pav.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]
