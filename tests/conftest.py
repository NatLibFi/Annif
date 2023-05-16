"""common fixtures for use by all test classes"""

import os.path
import shutil
import unittest.mock

import py.path
import pytest

import annif
import annif.analyzer
import annif.corpus
import annif.project
import annif.registry


@pytest.fixture(scope="module")
def app():
    # make sure the dummy vocab is in place because many tests depend on it
    subjfile = os.path.join(os.path.dirname(__file__), "corpora", "dummy-subjects.csv")
    app = annif.create_app(config_name="annif.default_config.TestingConfig")
    with app.app_context():
        project = annif.registry.get_project("dummy-en")
        # the vocab is needed for both English and Finnish language projects
        vocab = annif.corpus.SubjectFileCSV(subjfile)
        project.vocab.load_vocabulary(vocab)
    return app


@pytest.fixture(scope="module")
def app_with_initialize():
    app = annif.create_app(config_name="annif.default_config.TestingInitializeConfig")
    return app


@pytest.fixture
def app_client(app):
    with app.test_client() as app_client:
        yield app_client


@pytest.fixture(scope="module")
def registry(app):
    with app.app_context():
        return app.annif_registry


@pytest.fixture(scope="module")
def datadir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


@pytest.fixture(scope="module")
def testdatadir(app):
    """a fixture to access the tests/data directory as a py.path.local
    object"""
    with app.app_context():
        dir = py.path.local(app.config["DATADIR"])
    # clean up previous state of datadir
    shutil.rmtree(os.path.join(str(dir), "projects"), ignore_errors=True)
    return dir


@pytest.fixture(scope="module")
def subject_file():
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "subjects.tsv"
    )
    return annif.corpus.SubjectFileTSV(docfile, "fi")


@pytest.fixture(scope="module")
def dummy_subject_index(testdatadir):
    """a fixture to access the subject index of the dummy vocabulary"""
    vocab = annif.vocab.AnnifVocabulary("dummy", testdatadir)
    return vocab.subjects


@pytest.fixture(scope="module")
def vocabulary(datadir):
    vocab = annif.vocab.AnnifVocabulary("my-vocab", datadir)
    subjfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "yso-archaeology.ttl"
    )
    subjects = annif.corpus.SubjectFileSKOS(subjfile)
    vocab.load_vocabulary(subjects)
    return vocab


@pytest.fixture(scope="module")
def subject_index(vocabulary):
    return vocabulary.subjects


@pytest.fixture(scope="module")
def document_corpus(subject_index):
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    doc_corpus = annif.corpus.DocumentFile(docfile, subject_index)
    return doc_corpus


@pytest.fixture(scope="module")
def fulltext_corpus(subject_index):
    ftdir = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "fulltext"
    )
    ft_corpus = annif.corpus.DocumentDirectory(
        ftdir, subject_index, "fi", require_subjects=True
    )
    return ft_corpus


@pytest.fixture(scope="module")
def pretrained_vectors():
    return py.path.local(
        os.path.join(
            os.path.dirname(__file__), "corpora", "archaeology", "fasttext.vec"
        )
    )


@pytest.fixture(scope="module")
def project(subject_index, datadir, registry, vocabulary):
    proj = unittest.mock.Mock()
    proj.analyzer = annif.analyzer.get_analyzer("snowball(finnish)")
    proj.language = "fi"
    proj.vocab = vocabulary
    proj.subjects = subject_index
    proj.datadir = str(datadir)
    proj.registry = registry
    return proj


@pytest.fixture(scope="module")
def app_project(app):
    with app.app_context():
        dir = py.path.local(app.config["DATADIR"])
        shutil.rmtree(os.path.join(str(dir), "projects"), ignore_errors=True)
        return annif.registry.get_project("dummy-en")


@pytest.fixture(scope="function")
def empty_corpus(tmpdir, subject_index):
    empty_file = tmpdir.ensure("empty.tsv")
    return annif.corpus.DocumentFile(str(empty_file), subject_index)
