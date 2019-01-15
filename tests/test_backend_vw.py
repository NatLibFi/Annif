"""Unit tests for the fastText backend in Annif"""

import pytest
import annif.backend
import annif.corpus

pytest.importorskip("annif.backend.vw")


def test_vw_train(datadir, document_corpus, project):
    vw_type = annif.backend.get_backend("vw")
    vw = vw_type(
        backend_id='vw',
        params={'chunksize': 4},
        datadir=str(datadir))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_train_unknown_subject(tmpdir, datadir, project):
    vw_type = annif.backend.get_backend("vw")
    vw = vw_type(
        backend_id='vw',
        params={'chunksize': 4},
        datadir=str(datadir))

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("nonexistent\thttp://example.com/nonexistent\n" +
                  "arkeologia\thttp://www.yso.fi/onto/yso/p1265")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    vw.train(document_corpus, project)
    assert vw._model is not None
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_analyze(datadir, project):
    vw_type = annif.backend.get_backend("vw")
    vw = vw_type(
        backend_id='vw',
        params={'chunksize': 4},
        datadir=str(datadir))

    results = vw.analyze("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]
