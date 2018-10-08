"""Unit tests for projects in Annif"""

import pytest
import annif.project
import annif.backend.dummy


def test_get_project_en(app):
    with app.app_context():
        project = annif.project.get_project('dummy-en')
    assert project.project_id == 'dummy-en'
    assert project.language == 'en'
    assert project.analyzer.name == 'snowball'
    assert project.analyzer.param == 'english'
    assert len(project.backends) == 1
    assert isinstance(project.backends[0][0], annif.backend.dummy.DummyBackend)
    assert project.backends[0][1] == 0.5


def test_get_project_fi(app):
    with app.app_context():
        project = annif.project.get_project('dummy-fi')
    assert project.project_id == 'dummy-fi'
    assert project.language == 'fi'
    assert project.analyzer.name == 'snowball'
    assert project.analyzer.param == 'finnish'
    assert len(project.backends) == 1
    assert isinstance(project.backends[0][0], annif.backend.dummy.DummyBackend)
    assert project.backends[0][1] == 1.0


def test_get_project_fi_dump(app):
    with app.app_context():
        project = annif.project.get_project('dummy-fi')
    pdump = project.dump()
    assert pdump == {
        'project_id': 'dummy-fi',
        'name': 'Dummy Finnish',
        'language': 'fi',
        'backends': [{
            'backend_id': 'dummy',
            'weight': 1.0
        }]
    }


def test_get_project_nonexistent(app):
    with app.app_context():
        with pytest.raises(ValueError):
            annif.project.get_project('nonexistent')


def test_project_load_vocabulary(app, vocabulary, testdatadir):
    with app.app_context():
        project = annif.project.get_project('fasttext-fi')
    project.load_vocabulary(vocabulary)
    assert testdatadir.join('projects/fasttext-fi/subjects').exists()
    assert testdatadir.join('projects/fasttext-fi/subjects').size() > 0


def test_project_load_documents(app, document_corpus, testdatadir):
    with app.app_context():
        project = annif.project.get_project('fasttext-fi')
    project.load_documents(document_corpus)
    assert testdatadir.join('projects/fasttext-fi/fasttext-model').exists()
    assert testdatadir.join('projects/fasttext-fi/fasttext-model').size() > 0


def test_project_analyze(app):
    with app.app_context():
        project = annif.project.get_project('dummy-en')
    result = project.analyze('this is some text')
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0


def test_project_analyze_combine(app):
    with app.app_context():
        project = annif.project.get_project('dummydummy')
    result = project.analyze('this is some text')
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0


def test_project_not_initialized(app):
    with app.app_context():
        project = annif.project.get_project('dummy-en')
    assert not project.initialized
    dummy = project.backends[0][0]
    assert not dummy.initialized


def test_project_initialized(app_with_initialize):
    with app_with_initialize.app_context():
        project = annif.project.get_project('dummy-en')
    assert project.initialized
    dummy = project.backends[0][0]
    assert dummy.initialized
