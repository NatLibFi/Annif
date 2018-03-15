"""Unit test module for Annif operations"""

import random
import annif.operations


# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
    random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))


def test_list_projects():
    projects = annif.operations.list_projects()
    assert len(projects) > 0


def test_show_project():
    assert annif.operations.show_project(TEMP_PROJECT) is None
    myproj = annif.operations.show_project('myproject-fi')
    assert myproj is not None


def test_list_subjects():
    pass


def test_show_subject():
    pass


def test_load():
    pass


def test_drop_subject():
    pass


def test_analyze():
    pass
