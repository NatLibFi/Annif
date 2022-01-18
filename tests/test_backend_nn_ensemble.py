"""Unit tests for the nn_ensemble backend in Annif"""

import time
import pytest
import py.path
from datetime import datetime, timedelta, timezone
import annif.backend
import annif.corpus
from annif.exception import NotInitializedException, NotSupportedException

pytest.importorskip("annif.backend.nn_ensemble")
lmdb = pytest.importorskip("lmdb")


def test_lmdb_idx_to_key_to_idx():
    assert annif.backend.nn_ensemble.idx_to_key(42) == b'00000042'
    assert annif.backend.nn_ensemble.key_to_idx(b'00000042') == 42


def test_nn_ensemble_initialize_parallel(project):
    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=project)

    nn_ensemble.initialize(parallel=True)
    # model is still not loaded since we're preparing for parallel execution
    assert nn_ensemble._model is None


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


def test_nn_ensemble_can_set_lr(registry):
    project = registry.get_project('dummy-en')
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'epochs': 1, 'lr': 0.002},
        project=project)
    nn_ensemble._create_model(['dummy-en'])
    assert nn_ensemble._model.optimizer.learning_rate.value() == 0.002


def test_set_lmdb_map_size(registry, tmpdir):
    project = registry.get_project('dummy-en')
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en', 'epochs': 1, 'lmdb_map_size': 1},
        project=project)
    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none\n" * 40)
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    with pytest.raises(lmdb.MapFullError):
        nn_ensemble.train(document_corpus)


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

    # check adam default learning_rate:
    assert nn_ensemble._model.optimizer.learning_rate.value() == 0.001

    datadir = py.path.local(project.datadir)
    assert datadir.join('nn-model.h5').exists()
    assert datadir.join('nn-model.h5').size() > 0

    # test online learning
    modelfile = datadir.join('nn-model.h5')

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    time.sleep(0.1)  # make sure the timestamp has a chance to increase

    # Learning is typically performed on one document at a time
    document_corpus_single_doc = annif.corpus.LimitingDocumentCorpus(
        document_corpus, 1)
    nn_ensemble.learn(document_corpus_single_doc)

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
    assert datetime.now(timezone.utc) - \
        nn_ensemble.modification_time < timedelta(1)


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


def test_nn_ensemble_default_params(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=app_project)

    expected_default_params = {
        'optimizer': 'adam',
        'limit': 100,
    }
    actual_params = nn_ensemble.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_nn_ensemble_train_nodocuments(project, empty_corpus):
    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        config_params={'sources': 'dummy-en'},
        project=project)
    with pytest.raises(NotSupportedException):
        nn_ensemble.train(empty_corpus)


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
