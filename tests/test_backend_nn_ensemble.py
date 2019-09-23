"""Unit tests for the nn_ensemble backend in Annif"""

import time
import pytest
import annif.backend
import annif.corpus
import annif.project
from annif.exception import NotInitializedException

pytest.importorskip("annif.backend.nn_ensemble")


def test_nn_ensemble_suggest_no_model(datadir, project):
    nn_ensemble_type = annif.backend.get_backend('nn_ensemble')
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    with pytest.raises(NotInitializedException):
        results = nn_ensemble.suggest("example text", project)


def test_nn_ensemble_train_and_learn(app, datadir, tmpdir):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id='nn_ensemble',
        params={'sources': 'dummy-en'},
        datadir=str(datadir))

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))
    project = annif.project.get_project('dummy-en')

    with app.app_context():
        nn_ensemble.train(document_corpus, project)
    assert datadir.join('nn-model.h5').exists()
    assert datadir.join('nn-model.h5').size() > 0

    # test online learning
    modelfile = datadir.join('nn-model.h5')

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    time.sleep(0.1)  # make sure the timestamp has a chance to increase

    nn_ensemble.learn(document_corpus, project)

    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime
