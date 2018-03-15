"""Unit test module for Annif CLI commands"""

import random
import re
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


def test_eval_label(tmpdir):
    keyfile = tmpdir.join('dummy.key')
    keyfile.write("dummy\nanother\n")

    result = runner.invoke(
        annif.cli.run_eval, [
            'myproject-en', str(keyfile)], input='nothing special')
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search('Precision:\s+(\d.\d+)', result.output)
    print(precision.group(1))
    assert float(precision.group(1)) == 1.0
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search('F-measure:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) > 0.66
    assert float(f_measure.group(1)) < 0.67


def test_eval_uri(tmpdir):
    keyfile = tmpdir.join('dummy.key')
    keyfile.write(
        "<http://example.org/one>\tone\n<http://example.org/dummy>\tdummy\n")

    result = runner.invoke(
        annif.cli.run_eval, [
            'myproject-en', str(keyfile)], input='nothing special')
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search('Precision:\s+(\d.\d+)', result.output)
    print(precision.group(1))
    assert float(precision.group(1)) == 1.0
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search('F-measure:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) > 0.66
    assert float(f_measure.group(1)) < 0.67
