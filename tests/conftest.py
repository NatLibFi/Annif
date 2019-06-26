"""common fixtures for use by all test classes"""

import os.path
import shutil
import pytest
import py.path
import unittest.mock
import annif


@pytest.fixture(scope='module')
def app():
    # make sure the dummy vocab is in place because many tests depend on it
    subjfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'dummy-subjects.tsv')
    vocab = annif.corpus.SubjectFileTSV(subjfile)
    app = annif.create_app(config_name='annif.default_config.TestingConfig')
    with app.app_context():
        project = annif.project.get_project('dummy-en')
        project.vocab.load_vocabulary(vocab)
    return app


@pytest.fixture(scope='module')
def app_with_initialize():
    app = annif.create_app(
        config_name='annif.default_config.TestingInitializeConfig')
    return app


@pytest.fixture(scope='module')
def datadir(tmpdir_factory):
    return tmpdir_factory.mktemp('data')


@pytest.fixture(scope='module')
def testdatadir(app):
    """a fixture to access the tests/data directory as a py.path.local
     object"""
    with app.app_context():
        dir = py.path.local(app.config['DATADIR'])
    # clean up previous state of datadir
    shutil.rmtree(os.path.join(str(dir), 'projects'), ignore_errors=True)
    return dir


@pytest.fixture(scope='module')
def vocabulary():
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects.tsv')
    return annif.corpus.SubjectFileTSV(docfile)


@pytest.fixture(scope='module')
def subject_index(vocabulary):
    return annif.corpus.SubjectIndex(vocabulary)


@pytest.fixture(scope='module')
def document_corpus(subject_index):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    doc_corpus = annif.corpus.DocumentFile(docfile)
    doc_corpus.set_subject_index(subject_index)
    return doc_corpus


@pytest.fixture(scope='module')
def project(document_corpus):
    proj = unittest.mock.Mock()
    proj.analyzer = annif.analyzer.get_analyzer('snowball(finnish)')
    proj.subjects = annif.corpus.SubjectIndex(document_corpus)
    return proj
