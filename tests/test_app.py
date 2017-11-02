#!/usr/bin/env python3

import sys
import os
import random
import connexion
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from click.testing import CliRunner
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import app

es = Elasticsearch()
index = IndicesClient(es)
runner = CliRunner()

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
        random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))
TEMP_INDEX = TEMP_PROJECT[:7]

# Set a random name for the project index, so as not to mess up possible
# production indices.
app.annif.app.config['INDEX_NAME'] = TEMP_INDEX


def test_init():
    name = app.annif.app.config['INDEX_NAME']
    # assert runner.invoke(app.run_init).exit_code == 0
    assert index.exists(name) == False
    result = app.init()
    assert 'Initialized' in result
    i = es.indices.get_alias("*").keys()
    assert index.exists(name)


def test_list_projects():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(app.run_list_projects, ['moi']).exit_code != 0
    assert runner.invoke(
            app.run_list_projects, ['moi', '--debug', 'y']).exit_code != 0
    result = runner.invoke(app.run_list_projects)
    assert result.exit_code == 0


def test_create_project():
    # assert not index.exists(annif.format_index_name(TEMP_PROJECT))
    # result = runner.invoke(annif.create_project,
    #                        [TEMP_PROJECT, '--language', 'en', '--analyzer',
    #                         'english'])
    # assert index.exists(annif.format_index_name(TEMP_PROJECT))
    # assert result.exit_code == 0
    # # Creating a project should not succeed if an insufficient amount of args
    # # are provided.
    # FAILED_PROJECT = 'wow'
    # assert not index.exists(annif.format_index_name(FAILED_PROJECT))
    # result = runner.invoke(annif.create_project,
    #                        [FAILED_PROJECT, '--language', 'en'])
    # assert not index.exists(annif.format_index_name(FAILED_PROJECT))
    # result = runner.invoke(annif.create_project,
    #                        [FAILED_PROJECT, '--analyzer', 'english'])
    # assert not index.exists(annif.format_index_name(FAILED_PROJECT))
    pass


def test_show_project():
    # result = runner.invoke(annif.show_project, [TEMP_PROJECT])
    # assert result.exit_code == 0
    # # Test should not fail even if the user queries for a non-existent project.
    # failed_result = runner.invoke(annif.show_project, ['nonexistent'])
    # assert failed_result.exit_code == 0
    pass


def test_drop_project():
    # assert index.exists(app.format_index_name(TEMP_PROJECT))
    # # result = runner.invoke(annif.drop_project, [TEMP_PROJECT])
    # result = app.drop_project(TEMP_PROJECT)
    # assert not index.exists(app.format_index_name(TEMP_PROJECT))
    # assert result.exit_code == 0
    pass


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
