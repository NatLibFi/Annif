"""Unit tests for projects in Annif"""

import pytest
import annif.project


def test_get_project():
    project = annif.project.get_project('myproject-en')
    assert project.project_id == 'myproject-en'
    assert project.language == 'en'
    assert project.analyzer == 'english'


def test_get_project_nonexistent():
    with pytest.raises(ValueError):
        annif.project.get_project('nonexistent')
