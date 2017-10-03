#!/usr/bin/env python3

import sys
import os
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from click.testing import CliRunner
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from annif import annif

es = Elasticsearch()
index = IndicesClient(es)
runner = CliRunner()


# A dummy test for setup purposes
def test_start():
    assert annif.start() == "Started application"


def test_init():
    result = runner.invoke(annif.init)
    assert index.exists('annif')  # TODO: read index name from configuration
    assert result.exit_code == 0


def test_listprojects():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(annif.listprojects, ['moi']).exit_code != 0
    assert runner.invoke(
            annif.listprojects, ['moi', '--debug', 'y']).exit_code != 0


def test_showProject():
    pass


def test_createProject():
    pass


def test_dropProject():
    pass


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
