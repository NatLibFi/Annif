"""Unit test module for Annif CLI commands"""

import random
from click.testing import CliRunner
import annif.cli

runner = CliRunner()

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
    random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))


def test_list_projects():
    result = runner.invoke(annif.cli.run_list_projects)
    assert not result.exception
    assert result.exit_code == 0


def test_list_projects_bad_arguments():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(annif.cli.run_list_projects, ['moi']).exit_code != 0
    assert runner.invoke(
        annif.cli.run_list_projects, ['moi', '--debug', 'y']).exit_code != 0


def test_show_project():
    assert runner.invoke(
        annif.cli.run_show_project,
        [TEMP_PROJECT]).exit_code != 0
    # Test should not fail even if the user queries for a non-existent project.
    failed_result = runner.invoke(annif.cli.run_show_project, ['nonexistent'])
    assert failed_result.exception


def test_list_subjects():
    pass


def test_show_subject():
    pass


def test_load():
    pass


def test_drop_subject():
    pass


def test_analyze():
    result = runner.invoke(
        annif.cli.run_analyze,
        ['myproject-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "1.0\t<http://example.org/dummy>\tdummy\n"
    assert result.exit_code == 0


def test_analyze_nonexistent():
    result = runner.invoke(
        annif.cli.run_analyze,
        [TEMP_PROJECT],
        input='kissa')
    assert result.exception
    assert result.output == "No projects found with id '{}'.\n".format(
        TEMP_PROJECT)
    assert result.exit_code != 0
