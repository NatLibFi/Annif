#!/usr/bin/env python3

import sys
import os
import random
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from click.testing import CliRunner
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # Operate from root
import annif

es = Elasticsearch()
index = IndicesClient(es)
runner = CliRunner()

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
        random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))
TEMP_INDEX = TEMP_PROJECT[:7]

# Set a random name for the project index, so as not to mess up possible
# production indices.
annif.annif.app.config['INDEX_NAME'] = TEMP_INDEX


def test_init():
    name = annif.annif.app.config['INDEX_NAME']
    # assert runner.invoke(annif.run_init).exit_code == 0
    assert not index.exists(name)
    result = annif.init()
    assert 'Initialized' in result
    # print(es.indices.get_alias("*").keys())
    assert index.exists(name)


def test_run_create_project():
    assert not index.exists(annif.format_index_name(TEMP_PROJECT))
    result = annif.create_project(TEMP_PROJECT, 'swahili', 'norwegian')
    print(result)
    assert index.exists(annif.format_index_name(TEMP_PROJECT))
    # Creating a project should not succeed if an insufficient amount of args
    # are provided.
    FAILED_PROJECT = 'wow'
    assert not index.exists(annif.format_index_name(FAILED_PROJECT))
    result = runner.invoke(annif.run_create_project,
                           [FAILED_PROJECT, '--language', 'en'])
    assert not index.exists(annif.format_index_name(FAILED_PROJECT))
    result = runner.invoke(annif.run_create_project,
                           [FAILED_PROJECT, '--analyzer', 'english'])
    assert not index.exists(annif.format_index_name(FAILED_PROJECT))
    assert not result.exception


def test_list_projects():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(annif.run_list_projects, ['moi']).exit_code != 0
    assert runner.invoke(
            annif.run_list_projects, ['moi', '--debug', 'y']).exit_code != 0
    result = runner.invoke(annif.run_list_projects)
    assert result.exit_code == 0


def test_show_project():
    assert runner.invoke(annif.run_show_project, [TEMP_PROJECT]).exit_code == 0
    assert annif.show_project(TEMP_PROJECT)
    # Test should not fail even if the user queries for a non-existent project.
    failed_result = runner.invoke(annif.run_show_project, ['nonexistent'])
    assert not failed_result.exception


def test_drop_project():
    assert index.exists(annif.format_index_name(TEMP_PROJECT))
    # result = runner.invoke(annif.run_drop_project, [TEMP_PROJECT])
    result = annif.drop_project(TEMP_PROJECT)
    assert not index.exists(annif.format_index_name(TEMP_PROJECT))


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
