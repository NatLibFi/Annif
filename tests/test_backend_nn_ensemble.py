"""Unit tests for the nn_ensemble backend in Annif"""

import time
import pytest
import py.path
from datetime import datetime, timedelta
import annif.backend
import annif.corpus
from annif.exception import NotInitializedException

pytest.importorskip("annif.backend.nn_ensemble")


def test_lmdb_idx_to_key_to_idx():
    assert annif.backend.nn_ensemble.idx_to_key(42) == b'00000042'
    assert annif.backend.nn_ensemble.key_to_idx(b'00000042') == 42


def test_nn_ensemble_suggest_no_model(project):
    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=project)

    with pytest.raises(NotInitializedException):
        nn_ensemble.suggest("example text")


def test_nn_ensemble_is_not_trained(app_project):
    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=app_project)
    assert not nn_ensemble.is_trained


def test_nn_ensemble_train_and_learn(registry, tmpdir):
    project = registry.get_project('dummy-en')
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en', 'epochs': 1},
        project=project)

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none\n" * 40)
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    nn_ensemble.train(document_corpus)

    datadir = py.path.local(project.datadir)
    assert datadir.join('nn-model.h5').exists()
    assert datadir.join('nn-model.h5').size() > 0

    # test online learning
    modelfile = datadir.join('nn-model.h5')

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    time.sleep(0.1)  # make sure the timestamp has a chance to increase

    nn_ensemble.learn(document_corpus)

    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_nn_ensemble_train_cached(registry):
    # make sure we have the cached training data from the previous test
    project = registry.get_project('dummy-en')
    datadir = py.path.local(project.datadir)
    assert datadir.join('nn-train.mdb').exists()

    datadir.join('nn-model.h5').remove()

    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en', 'epochs': 2},
        project=project)

    nn_ensemble.train("cached")

    assert datadir.join('nn-model.h5').exists()
    assert datadir.join('nn-model.h5').size() > 0


def test_nn_ensemble_train_and_learn_params(registry, tmpdir, capfd):
    project = registry.get_project('dummy-en')
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en', 'epochs': 3},
        project=project)

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    train_params = {'epochs': 3}
    nn_ensemble.train(document_corpus, train_params)
    out, _ = capfd.readouterr()
    assert 'Epoch 3/3' in out

    learn_params = {'learn-epochs': 2}
    nn_ensemble.learn(document_corpus, learn_params)
    out, _ = capfd.readouterr()
    assert 'Epoch 2/2' in out


def test_nn_ensemble_is_trained(app_project):
    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=app_project)
    assert nn_ensemble.is_trained


def test_nn_ensemble_modification_time(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=app_project)
    assert datetime.now() - nn_ensemble.modification_time < timedelta(1)


def test_nn_ensemble_initialize(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=app_project)

    assert nn_ensemble._model is None
    nn_ensemble.initialize()
    assert nn_ensemble._model is not None
    # initialize a second time - this shouldn't do anything
    nn_ensemble.initialize()


def test_nn_ensemble_suggest(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=app_project)

    results = nn_ensemble.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen
        tiede tai oikeammin joukko tieteitä, jotka tutkivat ihmisen
        menneisyyttä. Tutkimusta tehdään analysoimalla muinaisjäännöksiä
        eli niitä jälkiä, joita ihmisten toiminta on jättänyt maaperään
        tai vesistöjen pohjaan.""")

    assert nn_ensemble._model is not None
    assert len(results) > 0
