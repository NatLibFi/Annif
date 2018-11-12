"""Unit tests for the PAV backend in Annif"""

import annif.backend


def test_pav_load_documents(app, datadir, document_corpus, project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 1, 'sources': 'dummy-fi'},
        datadir=str(datadir))

    with app.app_context():
        pav.load_corpus(document_corpus, project)
    assert datadir.join('pav-model-dummy-fi').exists()
    assert datadir.join('pav-model-dummy-fi').size() > 0


def test_pav_analyze(app, datadir, project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 1, 'sources': 'dummy-fi'},
        datadir=str(datadir))

    results = pav.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0
