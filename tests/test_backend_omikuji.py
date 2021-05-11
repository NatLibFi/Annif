"""Unit tests for the Omikuji backend in Annif"""

import pytest
import annif.backend
import annif.corpus
from annif.exception import NotInitializedException
from annif.exception import NotSupportedException

pytest.importorskip("annif.backend.omikuji")


def test_omikuji_default_params(project):
    omikuji_type = annif.backend.get_backend("omikuji")
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    expected_default_params = {
        'limit': 100,
        'min_df': 1,
    }
    actual_params = omikuji.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_omikuji_suggest_no_vectorizer(project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    with pytest.raises(NotInitializedException):
        omikuji.suggest("example text")


def test_omikuji_create_train_file(tmpdir, project, datadir):
    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("nonexistent\thttp://example.com/nonexistent\n" +
                  "arkeologia\thttp://www.yso.fi/onto/yso/p1265\n" +
                  "...\thttp://example.com/none")
    corpus = annif.corpus.DocumentFile(str(tmpfile))
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)
    input = (doc.text for doc in corpus.documents)
    veccorpus = omikuji.create_vectorizer(input, {})
    omikuji._create_train_file(veccorpus, corpus)
    assert datadir.join('omikuji-train.txt').exists()
    traindata = datadir.join('omikuji-train.txt').read().splitlines()
    assert len(traindata) == 2  # header + 1 example
    examples, features, labels = map(int, traindata[0].split())
    assert examples == 1
    assert features == 2
    assert labels == 130


def test_omikuji_train(datadir, document_corpus, project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    # verify that training works even if there is a preexisting model directory
    # - to simulate this we will create an empty directory instead
    datadir.join('omikuji-model').ensure(dir=True)
    assert not datadir.join('omikuji-model').listdir()  # empty dir

    omikuji.train(document_corpus)
    assert omikuji._model is not None
    assert datadir.join('omikuji-model').exists()
    assert datadir.join('omikuji-model').listdir()  # non-empty dir


def test_omikuji_train_ngram(datadir, document_corpus, project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={'ngram': 2},
        project=project)

    datadir.join('omikuji-model').remove()
    omikuji.train(document_corpus)
    assert omikuji._model is not None
    assert datadir.join('omikuji-model').exists()
    assert datadir.join('omikuji-model').listdir()  # non-empty dir


def test_omikuji_train_cached(datadir, project):
    assert datadir.join('omikuji-train.txt').exists()
    datadir.join('omikuji-model').remove()
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)
    omikuji.train("cached")
    assert omikuji._model is not None
    assert datadir.join('omikuji-model').exists()
    assert datadir.join('omikuji-model').listdir()  # non-empty dir


def test_omikuji_train_nodocuments(datadir, project, empty_corpus):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    with pytest.raises(NotSupportedException):
        omikuji.train(empty_corpus)


def test_omikuji_train_params(datadir, document_corpus, project, capfd):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)
    params = {'cluster_k': 1, 'max_depth': 2, 'collapse_every_n_layers': 42}
    omikuji.train(document_corpus, params)

    out, _ = capfd.readouterr()
    parameters_heading = 'Training model with hyper-parameters HyperParam'
    assert parameters_heading in out
    for line in out.splitlines():
        if parameters_heading in line:
            assert 'k: 1' in line
            assert 'max_depth: 2' in line
            assert 'collapse_every_n_layers: 42' in line


def test_omikuji_suggest(project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={'limit': 8},
        project=project)

    results = omikuji.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0
    assert len(results) <= 8
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in hits]
    assert 'arkeologia' in [result.label for result in hits]


def test_omikuji_suggest_no_input(project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={'limit': 8},
        project=project)

    results = omikuji.suggest("j")
    assert len(results) == 0


def test_omikuji_suggest_no_model(datadir, project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    datadir.join('omikuji-model').remove()
    with pytest.raises(NotInitializedException):
        omikuji.suggest("example text")
