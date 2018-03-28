"""Unit tests for projects in Annif"""

import pytest
import annif.project
import annif.backend.dummy


def test_get_project_en(app):
    with app.app_context():
        project = annif.project.get_project('dummy-en')
    assert project.project_id == 'dummy-en'
    assert project.language == 'en'
    assert len(project.backends) == 1
    assert isinstance(project.backends[0][0], annif.backend.dummy.DummyBackend)
    assert project.backends[0][1] == 0.5


def test_get_project_fi(app):
    with app.app_context():
        project = annif.project.get_project('dummy-fi')
    assert project.project_id == 'dummy-fi'
    assert project.language == 'fi'
    assert len(project.backends) == 1
    assert isinstance(project.backends[0][0], annif.backend.dummy.DummyBackend)
    assert project.backends[0][1] == 1.0


def test_get_project_fi_dump(app):
    with app.app_context():
        project = annif.project.get_project('dummy-fi')
    pdump = project.dump()
    assert pdump == {
        'project_id': 'dummy-fi',
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


def test_project_analyze(app):
    with app.app_context():
        project = annif.project.get_project('dummy-en')
    result = project.analyze('this is some text', limit=10, threshold=0.0)
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 0.5
