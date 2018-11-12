"""Unit tests for the fastText backend in Annif"""

import annif.backend
import annif.corpus


def test_fasttext_load_documents(datadir, document_corpus, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        params={
            'limit': 50,
            'dim': 100,
            'lr': 0.25,
            'epoch': 20,
            'loss': 'hs'},
        datadir=str(datadir))

    fasttext.load_corpus(document_corpus, project)
    assert fasttext._model is not None
    assert datadir.join('fasttext-model').exists()
    assert datadir.join('fasttext-model').size() > 0


def test_fasttext_load_documents_unknown_subject(tmpdir, datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        params={
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

    fasttext.load_corpus(document_corpus, project)
    assert fasttext._model is not None
    assert datadir.join('fasttext-model').exists()
    assert datadir.join('fasttext-model').size() > 0


def test_fasttext_analyze(datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id='fasttext',
        params={
            'limit': 50,
            'chunksize': 1,
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

    assert len(results) > 0
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]
