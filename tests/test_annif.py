#!/usr/bin/env python3

import sys
import os
import random
from click.testing import CliRunner
import annif
import annif.operations
import annif.cli

runner = CliRunner()

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))

def test_list_projects():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(annif.cli.run_list_projects, ['moi']).exit_code != 0
    assert runner.invoke(
            annif.cli.run_list_projects, ['moi', '--debug', 'y']).exit_code != 0
    result = runner.invoke(annif.cli.run_list_projects)
    assert result.exit_code == 0


def test_show_project():
    assert runner.invoke(annif.cli.run_show_project, [TEMP_PROJECT]).exit_code == 0
    assert annif.operations.show_project(TEMP_PROJECT)
    # Test should not fail even if the user queries for a non-existent project.
    failed_result = runner.invoke(annif.cli.run_show_project, ['nonexistent'])
    assert not failed_result.exception


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
