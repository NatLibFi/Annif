#!/usr/bin/env python3

import sys
import os
import click
import random
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from click.testing import CliRunner
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from annif import annif

es = Elasticsearch()
index = IndicesClient(es)
runner = CliRunner()

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))


# A dummy test for setup purposes
def test_start():
    assert annif.start() == "Started application"


def test_init():
    result = runner.invoke(annif.init)
    assert index.exists(annif.annif.config['INDEX_NAME'])
    assert result.exit_code == 0


def test_listprojects():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(annif.listprojects, ['moi']).exit_code != 0
    assert runner.invoke(
            annif.listprojects, ['moi', '--debug', 'y']).exit_code != 0
    result = runner.invoke(annif.listprojects)


def test_createProject():
    assert not index.exists(annif.parseIndexname(TEMP_PROJECT))
    result = runner.invoke(annif.createProject, [TEMP_PROJECT, '--language',
        'en', '--analyzer', 'english'])
    assert index.exists(annif.parseIndexname(TEMP_PROJECT))
    assert result.exit_code == 0


def test_showProject():
    result = runner.invoke(annif.showProject, [TEMP_PROJECT])
    assert result.exit_code == 0


def test_dropProject():
    assert index.exists(annif.parseIndexname(TEMP_PROJECT))
    result = runner.invoke(annif.dropProject, [TEMP_PROJECT])
    assert not index.exists(annif.parseIndexname(TEMP_PROJECT))
    assert result.exit_code == 0


def test_listSubjects():
    pass


def test_showSubject():
    pass


def createSubject():
    pass


def test_load():
    pass


def dropSubject():
    pass


def test_analyze():
    pass
