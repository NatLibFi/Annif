"""common fixtures for use by all test classes"""

import os.path
import shutil
import pytest
import py.path
import annif


@pytest.fixture(scope='session')
def app():
    app = annif.create_app(config_name='config.TestingConfig')
    return app


@pytest.fixture(scope='module')
def app_with_initialize():
    app = annif.create_app(config_name='config.TestingInitializeConfig')
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
    shutil.rmtree(str(dir), ignore_errors=True)
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
