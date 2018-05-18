"""common fixtures for use by all test classes"""

import os.path
import pytest
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
def subject_corpus():
    subjdir = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects')
    return annif.corpus.SubjectDirectory(subjdir)


@pytest.fixture(scope='module')
def document_corpus():
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    return annif.corpus.DocumentFile(docfile)
