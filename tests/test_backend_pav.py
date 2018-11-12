"""Unit tests for the PAV backend in Annif"""

import annif.backend
import annif.corpus


def test_pav_load_documents(app, datadir, tmpdir, project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        datadir=str(datadir))

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    with app.app_context():
        pav.load_corpus(document_corpus, project)
    assert datadir.join('pav-model-dummy-fi').exists()
    assert datadir.join('pav-model-dummy-fi').size() > 0


def test_pav_initialize(app, datadir):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        datadir=str(datadir))

    assert pav._models is None
    with app.app_context():
        pav.initialize()
    assert pav._models is not None
    # initialize a second time - this shouldn't do anything
    with app.app_context():
        pav.initialize()


def test_pav_analyze(app, datadir, project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        datadir=str(datadir))

    results = pav.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(pav._models['dummy-fi']) == 1
    assert len(results) > 0
