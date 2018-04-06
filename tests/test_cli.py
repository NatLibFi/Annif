"""Unit test module for Annif CLI commands"""

import random
import re
import os.path
import py.path
import pytest
from click.testing import CliRunner
import annif.cli

runner = CliRunner(env={'ANNIF_CONFIG': 'config.TestingConfig'})

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
    random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))


@pytest.fixture(scope='session')
def datadir(app):
    with app.app_context():
        dir = py.path.local(app.config['DATADIR'])
    return dir


def test_list_projects():
    result = runner.invoke(annif.cli.cli, ["list-projects"])
    assert not result.exception
    assert result.exit_code == 0


def test_list_projects_bad_arguments():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(
        annif.cli.cli, [
            'list-projects', 'moi']).exit_code != 0
    assert runner.invoke(
        annif.cli.run_list_projects, ['moi', '--debug', 'y']).exit_code != 0


def test_show_project():
    assert runner.invoke(
        annif.cli.cli,
        ['show-project', TEMP_PROJECT]).exit_code != 0
    # Test should not fail even if the user queries for a non-existent project.
    failed_result = runner.invoke(
        annif.cli.cli, [
            'show-project', 'nonexistent'])
    assert failed_result.exception


def test_list_subjects():
    pass


def test_show_subject():
    pass


def test_load(datadir):
    subjdir = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects')
    result = runner.invoke(annif.cli.cli, ['load', 'tfidf-fi', subjdir])
    assert not result.exception
    assert result.exit_code == 0
    assert datadir.join('projects/tfidf-fi/subjects').exists()
    assert datadir.join('projects/tfidf-fi/subjects').size() > 0
    assert datadir.join('projects/tfidf-fi/vectorizer').exists()
    assert datadir.join('projects/tfidf-fi/vectorizer').size() > 0
    assert datadir.join('backends/tfidf-fi/index').exists()
    assert datadir.join('backends/tfidf-fi/index').size() > 0


def test_drop_subject():
    pass


def test_analyze():
    result = runner.invoke(
        annif.cli.cli,
        ['analyze', 'dummy-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "1.0\t<http://example.org/dummy>\tdummy\n"
    assert result.exit_code == 0


def test_analyze_nonexistent():
    result = runner.invoke(
        annif.cli.cli,
        ['analyze', TEMP_PROJECT],
        input='kissa')
    assert result.exception
    assert result.output == "No projects found with id '{}'.\n".format(
        TEMP_PROJECT)
    assert result.exit_code != 0


def test_analyze_param():
    result = runner.invoke(
        annif.cli.cli,
        ['analyze', '--backend-param', 'dummy.score=0.8', 'dummy-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "0.8\t<http://example.org/dummy>\tdummy\n"
    assert result.exit_code == 0


def test_analyzedir(tmpdir):
    tmpdir.join('doc1.txt').write('nothing special')

    result = runner.invoke(
        annif.cli.cli, ['analyzedir', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    assert tmpdir.join('doc1.annif').exists()
    assert tmpdir.join('doc1.annif').read_text(
        'utf-8') == "<http://example.org/dummy>\tdummy\t0.5\n"


def test_eval_label(tmpdir):
    keyfile = tmpdir.join('dummy.key')
    keyfile.write("dummy\nanother\n")

    result = runner.invoke(
        annif.cli.cli, [
            'eval', 'dummy-en', str(keyfile)], input='nothing special')
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search('Precision:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 1.0
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search('F-measure:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) > 0.66
    assert float(f_measure.group(1)) < 0.67
    precision1 = re.search('Precision@1:\s+(\d.\d+)', result.output)
    assert float(precision1.group(1)) == 1.0
    precision3 = re.search('Precision@3:\s+(\d.\d+)', result.output)
    assert float(precision3.group(1)) == 1.0
    precision5 = re.search('Precision@5:\s+(\d.\d+)', result.output)
    assert float(precision5.group(1)) == 1.0
    true_positives = re.search('True positives:\s+(\d+)', result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search('False positives:\s+(\d+)', result.output)
    assert int(false_positives.group(1)) == 0
    false_negatives = re.search('False negatives:\s+(\d+)', result.output)
    assert int(false_negatives.group(1)) == 1


def test_eval_param(tmpdir):
    keyfile = tmpdir.join('dummy.key')
    keyfile.write("dummy\nanother\n")

    result = runner.invoke(annif.cli.cli,
                           ['eval',
                            '--backend-param',
                            'dummy.score=0.0',
                            'dummy-en',
                            str(keyfile)],
                           input='nothing special')
    assert not result.exception
    assert result.exit_code == 0

    # since zero scores were set with the parameter, there should be no hits
    # at all
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.0


def test_eval_uri(tmpdir):
    keyfile = tmpdir.join('dummy.key')
    keyfile.write(
        "<http://example.org/one>\tone\n<http://example.org/dummy>\tdummy\n")

    result = runner.invoke(
        annif.cli.cli, [
            'eval', 'dummy-en', str(keyfile)], input='nothing special')
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search('Precision:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 1.0
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search('F-measure:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) > 0.66
    assert float(f_measure.group(1)) < 0.67
    precision1 = re.search('Precision@1:\s+(\d.\d+)', result.output)
    assert float(precision1.group(1)) == 1.0
    precision3 = re.search('Precision@3:\s+(\d.\d+)', result.output)
    assert float(precision3.group(1)) == 1.0
    precision5 = re.search('Precision@5:\s+(\d.\d+)', result.output)
    assert float(precision5.group(1)) == 1.0
    true_positives = re.search('True positives:\s+(\d+)', result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search('False positives:\s+(\d+)', result.output)
    assert int(false_positives.group(1)) == 0
    false_negatives = re.search('False negatives:\s+(\d+)', result.output)
    assert int(false_negatives.group(1)) == 1


def test_evaldir(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(annif.cli.cli, ['evaldir', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search('Precision:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search('F-measure:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) == 0.5
    precision1 = re.search('Precision@1:\s+(\d.\d+)', result.output)
    assert float(precision1.group(1)) == 0.5
    precision3 = re.search('Precision@3:\s+(\d.\d+)', result.output)
    assert float(precision3.group(1)) == 0.5
    precision5 = re.search('Precision@5:\s+(\d.\d+)', result.output)
    assert float(precision5.group(1)) == 0.5
    true_positives = re.search('True positives:\s+(\d+)', result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search('False positives:\s+(\d+)', result.output)
    assert int(false_positives.group(1)) == 1
    false_negatives = re.search('False negatives:\s+(\d+)', result.output)
    assert int(false_negatives.group(1)) == 1


def test_evaldir_param(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(
        annif.cli.cli, [
            'evaldir', '--backend-param', 'dummy.score=0.0', 'dummy-en',
            str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    # since zero scores were set with the parameter, there should be no hits
    # at all
    recall = re.search('Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.0


def test_optimize(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(
        annif.cli.cli, [
            'optimize', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search('Best Precision:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search('Best Recall:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search('Best F-measure:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) == 0.5
