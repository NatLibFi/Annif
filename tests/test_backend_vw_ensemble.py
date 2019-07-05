"""Unit tests for the vw_ensemble backend in Annif"""

import json
import time
import pytest
import annif.backend
import annif.corpus
import annif.project
from annif.exception import NotInitializedException

pytest.importorskip("annif.backend.vw_ensemble")


def test_vw_ensemble_suggest_no_model(datadir, project):
    vw_ensemble_type = annif.backend.get_backend('vw_ensemble')
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    with pytest.raises(NotInitializedException):
        results = vw_ensemble.suggest("example text", project)


def test_vw_ensemble_train_and_learn(app, datadir, tmpdir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))
    project = annif.project.get_project('dummy-en')

    with app.app_context():
        vw_ensemble.train(document_corpus, project)
    assert datadir.join('vw-train.txt').exists()
    assert datadir.join('vw-train.txt').size() > 0
    assert datadir.join('subject-freq.json').exists()
    assert datadir.join('subject-freq.json').size() > 0
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0

    # test online learning
    modelfile = datadir.join('vw-model')
    freqfile = datadir.join('subject-freq.json')

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()
    with open(str(freqfile)) as freqf:
        old_totalfreq = sum(json.load(freqf).values())

    time.sleep(0.1)  # make sure the timestamp has a chance to increase

    vw_ensemble.learn(document_corpus, project)

    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime
    with open(str(freqfile)) as freqf:
        assert sum(json.load(freqf).values()) != old_totalfreq


def test_vw_ensemble_initialize(app, datadir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    assert vw_ensemble._model is None
    with app.app_context():
        vw_ensemble.initialize()
    assert vw_ensemble._model is not None
    # initialize a second time - this shouldn't do anything
    with app.app_context():
        vw_ensemble.initialize()


def test_vw_ensemble_suggest(app, datadir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    project = annif.project.get_project('dummy-en')

    results = vw_ensemble.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert vw_ensemble._model is not None
    assert len(results) > 0


def test_vw_ensemble_suggest_set_discount_rate(app, datadir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en', 'discount_rate': '0.02'},
        datadir=str(datadir))

    project = annif.project.get_project('dummy-en')

    results = vw_ensemble.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) > 0


def test_vw_ensemble_format_example(datadir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    ex = vw_ensemble._format_example(0, [0.5])
    assert ex == ' |0 dummy-en:0.500000'


def test_vw_ensemble_format_example_avoid_sci_notation(datadir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    ex = vw_ensemble._format_example(0, [7.24e-05])
    assert ex == ' |0 dummy-en:0.000072'
