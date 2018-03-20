"""Unit tests for projects in Annif"""

import pytest
import annif.project
import annif.backend.dummy


def test_get_project_en():
    project = annif.project.get_project('myproject-en')
    assert project.project_id == 'myproject-en'
    assert project.language == 'en'
    assert len(project.backends) == 1
    assert isinstance(project.backends[0][0], annif.backend.dummy.DummyBackend)
    assert project.backends[0][1] == 0.5


def test_get_project_fi():
    project = annif.project.get_project('myproject-fi')
    assert project.project_id == 'myproject-fi'
    assert project.language == 'fi'
    assert len(project.backends) == 1
    assert isinstance(project.backends[0][0], annif.backend.dummy.DummyBackend)
    assert project.backends[0][1] == 1.0


def test_get_project_fi_dump():
    project = annif.project.get_project('myproject-fi')
    pdump = project.dump()
    assert pdump == {
        'project_id': 'myproject-fi',
        'language': 'fi',
        'backends': [{
            'backend_id': 'dummy',
            'weight': 1.0
        }]
    }


def test_get_project_nonexistent():
    with pytest.raises(ValueError):
        annif.project.get_project('nonexistent')


def test_project_analyze():
    project = annif.project.get_project('myproject-en')
    result = project.analyze('this is some text', limit=10, threshold=0.0)
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 0.5
