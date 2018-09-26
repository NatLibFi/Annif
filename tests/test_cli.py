"""Unit test module for Annif CLI commands"""

import contextlib
import random
import re
import os.path
import shutil
import pytest
from click.testing import CliRunner
import annif.cli

runner = CliRunner(env={'ANNIF_CONFIG': 'config.TestingConfig'})

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
    random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))


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


def test_loadvoc_tsv(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    subjectfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects.tsv')
    result = runner.invoke(annif.cli.cli, ['loadvoc', 'tfidf-fi', subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/subjects').exists()
    assert testdatadir.join('projects/tfidf-fi/subjects').size() > 0


def test_loadvoc_rdf(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    subjectfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'yso-archaeology.rdf')
    result = runner.invoke(annif.cli.cli, ['loadvoc', 'tfidf-fi', subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/subjects').exists()
    assert testdatadir.join('projects/tfidf-fi/subjects').size() > 0


def test_loaddocs(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli, ['loaddocs', 'tfidf-fi', docfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/vectorizer').exists()
    assert testdatadir.join('projects/tfidf-fi/vectorizer').size() > 0
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').exists()
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').size() > 0


def test_loaddocs_multiple(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli,
                           ['loaddocs', 'tfidf-fi', docfile, docfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/vectorizer').exists()
    assert testdatadir.join('projects/tfidf-fi/vectorizer').size() > 0
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').exists()
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').size() > 0


def test_load(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/vectorizer')))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/tfidf-index')))
    subjdir = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects')
    result = runner.invoke(annif.cli.cli, ['load', 'tfidf-fi', subjdir])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/subjects').exists()
    assert testdatadir.join('projects/tfidf-fi/subjects').size() > 0
    assert testdatadir.join('projects/tfidf-fi/vectorizer').exists()
    assert testdatadir.join('projects/tfidf-fi/vectorizer').size() > 0
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').exists()
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').size() > 0


def test_analyze():
    result = runner.invoke(
        annif.cli.cli,
        ['analyze', 'dummy-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t1.0\n"
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
    assert result.output == "<http://example.org/dummy>\tdummy\t0.8\n"
    assert result.exit_code == 0


def test_analyze_ensemble():
    result = runner.invoke(
        annif.cli.cli,
        ['analyze', 'ensemble'],
        input='the cat sat on the mat')
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t1.0\n"
    assert result.exit_code == 0


def test_analyzedir(tmpdir):
    tmpdir.join('doc1.txt').write('nothing special')

    result = runner.invoke(
        annif.cli.cli, ['analyzedir', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    assert tmpdir.join('doc1.annif').exists()
    assert tmpdir.join('doc1.annif').read_text(
        'utf-8') == "<http://example.org/dummy>\tdummy\t1.0\n"

    # make sure that preexisting subject files are not overwritten
    result = runner.invoke(
        annif.cli.cli, ['analyzedir', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0
    assert "Not overwriting" in result.output

    # check that the --force parameter forces overwriting
    result = runner.invoke(
        annif.cli.cli, ['analyzedir', 'dummy-fi', '--force', str(tmpdir)])
    assert tmpdir.join('doc1.annif').exists()
    assert "Not overwriting" not in result.output
    assert tmpdir.join('doc1.annif').read_text(
        'utf-8') == "<http://example.org/dummy>\tdummy\t1.0\n"


def test_eval_label(tmpdir, testdatadir):
    keyfile = tmpdir.join('dummy.key')
    keyfile.write("dummy\nanother\n")
    subjectfile = testdatadir.ensure('projects/dummy-en/subjects')
    subjectfile.write("<http://example.org/dummy>\tdummy\n" +
                      "<http://example.org/none>\tnone\n")

    result = runner.invoke(
        annif.cli.cli, [
            'eval', 'dummy-en', str(keyfile)], input='nothing special')
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search(r'Precision .*doc.*:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 1.0
    recall = re.search(r'Recall .*doc.*:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r'F1 score .*doc.*:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) > 0.66
    assert float(f_measure.group(1)) < 0.67
    precision1 = re.search(r'Precision@1:\s+(\d.\d+)', result.output)
    assert float(precision1.group(1)) == 1.0
    precision3 = re.search(r'Precision@3:\s+(\d.\d+)', result.output)
    assert float(precision3.group(1)) == 1.0
    precision5 = re.search(r'Precision@5:\s+(\d.\d+)', result.output)
    assert float(precision5.group(1)) == 1.0
    lrap = re.search(r'LRAP:\s+(\d.\d+)', result.output)
    assert float(lrap.group(1)) == 1.0
    true_positives = re.search(r'True positives:\s+(\d+)', result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search(r'False positives:\s+(\d+)', result.output)
    assert int(false_positives.group(1)) == 0
    false_negatives = re.search(r'False negatives:\s+(\d+)', result.output)
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
    recall = re.search(r'Recall .*doc.*:\s+(\d.\d+)', result.output)
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

    precision = re.search(r'Precision .*doc.*:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 1.0
    recall = re.search(r'Recall.*doc.*:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r'F1 score .*doc.*:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) > 0.66
    assert float(f_measure.group(1)) < 0.67
    precision1 = re.search(r'Precision@1:\s+(\d.\d+)', result.output)
    assert float(precision1.group(1)) == 1.0
    precision3 = re.search(r'Precision@3:\s+(\d.\d+)', result.output)
    assert float(precision3.group(1)) == 1.0
    precision5 = re.search(r'Precision@5:\s+(\d.\d+)', result.output)
    assert float(precision5.group(1)) == 1.0
    lrap = re.search(r'LRAP:\s+(\d.\d+)', result.output)
    assert float(lrap.group(1)) == 1.0
    true_positives = re.search(r'True positives:\s+(\d+)', result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search(r'False positives:\s+(\d+)', result.output)
    assert int(false_positives.group(1)) == 0
    false_negatives = re.search(r'False negatives:\s+(\d+)', result.output)
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

    precision = re.search(r'Precision .*doc.*:\s+(\d.\d+)', result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search(r'Recall .*doc.*:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r'F1 score .*doc.*:\s+(\d.\d+)', result.output)
    assert float(f_measure.group(1)) == 0.5
    precision1 = re.search(r'Precision@1:\s+(\d.\d+)', result.output)
    assert float(precision1.group(1)) == 0.5
    precision3 = re.search(r'Precision@3:\s+(\d.\d+)', result.output)
    assert float(precision3.group(1)) == 0.5
    precision5 = re.search(r'Precision@5:\s+(\d.\d+)', result.output)
    assert float(precision5.group(1)) == 0.5
    lrap = re.search(r'LRAP:\s+(\d.\d+)', result.output)
    assert float(lrap.group(1)) == 0.75
    true_positives = re.search(r'True positives:\s+(\d+)', result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search(r'False positives:\s+(\d+)', result.output)
    assert int(false_positives.group(1)) == 1
    false_negatives = re.search(r'False negatives:\s+(\d+)', result.output)
    assert int(false_negatives.group(1)) == 1
    ndocs = re.search(r'Documents evaluated:\s+(\d+)', result.output)
    assert int(ndocs.group(1)) == 2


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
    recall = re.search(r'Recall .*doc.*:\s+(\d.\d+)', result.output)
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

    precision = re.search(r'Best Precision .*?doc.*?:\s+(\d.\d+)',
                          result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search(r'Best Recall .*?doc.*?:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r'Best F1 score .*?doc.*?:\s+(\d.\d+)',
                          result.output)
    assert float(f_measure.group(1)) == 0.5
    ndocs = re.search(r'Documents evaluated:\s+(\d)', result.output)
    assert int(ndocs.group(1)) == 2
