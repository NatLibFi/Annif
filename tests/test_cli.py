"""Unit test module for Annif CLI commands"""

import contextlib
import random
import re
import os.path
import pytest
import pkg_resources
from click.testing import CliRunner
import annif.cli

runner = CliRunner(env={'ANNIF_CONFIG': 'annif.default_config.TestingConfig'})

# Generate a random project name to use in tests
TEMP_PROJECT = ''.join(
    random.choice('abcdefghiklmnopqrstuvwxyz') for _ in range(8))
PROJECTS_FILE_OPTION = 'tests/projects_for_config_path_option.cfg'


def test_list_projects():
    result = runner.invoke(annif.cli.cli, ["list-projects"])
    assert not result.exception
    assert result.exit_code == 0
    # public project should be visible
    assert 'dummy-fi' in result.output
    # hidden project should be visible
    assert 'dummy-en' in result.output
    # private project should be visible
    assert 'dummydummy' in result.output
    # project with no access setting should be visible
    assert 'ensemble' in result.output


def test_list_projects_bad_arguments():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(
        annif.cli.cli, [
            'list-projects', 'moi']).exit_code != 0
    assert runner.invoke(
        annif.cli.run_list_projects, ['moi', '--debug', 'y']).exit_code != 0


def test_list_projects_config_path_option():
    result = runner.invoke(
        annif.cli.cli, ["list-projects", "--projects", PROJECTS_FILE_OPTION])
    assert not result.exception
    assert result.exit_code == 0
    assert 'dummy_for_projects_option' in result.output
    assert 'dummy-fi' not in result.output
    assert 'dummy-en' not in result.output


def test_list_projects_config_path_option_nonexistent():
    failed_result = runner.invoke(
        annif.cli.cli, ["list-projects", "--projects", "nonexistent.cfg"])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Error: Invalid value for '-p' / '--projects': " \
           "File 'nonexistent.cfg' does not exist." in failed_result.output


def test_show_project():
    result = runner.invoke(annif.cli.cli, ['show-project', 'dummy-en'])
    assert not result.exception

    project_id = re.search(r'Project ID:\s+(.+)', result.output)
    assert project_id.group(1) == 'dummy-en'
    project_name = re.search(r'Project Name:\s+(.+)', result.output)
    assert project_name.group(1) == 'Dummy English'
    project_lang = re.search(r'Language:\s+(.+)', result.output)
    assert project_lang.group(1) == 'en'
    access = re.search(r'Access:\s+(.+)', result.output)
    assert access.group(1) == 'hidden'
    is_trained = re.search(r'Trained:\s+(.+)', result.output)
    assert is_trained.group(1) == 'True'
    modification_time = re.search(r'Modification time:\s+(.+)', result.output)
    assert modification_time.group(1) == 'None'


def test_show_project_nonexistent():
    assert runner.invoke(
        annif.cli.cli,
        ['show-project', TEMP_PROJECT]).exit_code != 0
    # Test should not fail even if the user queries for a non-existent project.
    failed_result = runner.invoke(
        annif.cli.cli, [
            'show-project', 'nonexistent'])
    assert failed_result.exception


def test_clear_project(testdatadir):
    dirpath = os.path.join(str(testdatadir), 'projects', 'dummy-fi')
    fpath = os.path.join(str(dirpath), 'test_clear_project_datafile')
    os.makedirs(dirpath)
    open(fpath, 'a').close()

    assert runner.invoke(
        annif.cli.cli,
        ['clear', 'dummy-fi']).exit_code == 0
    assert not os.path.isdir(dirpath)


def test_clear_project_nonexistent_data(testdatadir, caplog):
    logger = annif.logger
    logger.propagate = True
    result = runner.invoke(annif.cli.cli, ['clear', 'dummy-fi'])
    assert not result.exception
    assert result.exit_code == 0
    assert len(caplog.records) == 1
    expected_msg = 'No model data to remove for project dummy-fi.'
    assert expected_msg == caplog.records[0].message


def test_loadvoc_tsv(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects.ttl')))
    subjectfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects.tsv')
    result = runner.invoke(annif.cli.cli, ['loadvoc', 'tfidf-fi', subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('vocabs/yso-fi/subjects').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects').size() > 0
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').size() > 0


def test_loadvoc_tsv_with_bom(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects.ttl')))
    subjectfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects-bom.tsv')
    result = runner.invoke(annif.cli.cli, ['loadvoc', 'tfidf-fi', subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('vocabs/yso-fi/subjects').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects').size() > 0
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').size() > 0


def test_loadvoc_rdf(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects.ttl')))
    subjectfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'yso-archaeology.rdf')
    result = runner.invoke(annif.cli.cli, ['loadvoc', 'tfidf-fi', subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('vocabs/yso-fi/subjects').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects').size() > 0
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').size() > 0


def test_loadvoc_ttl(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects')))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join('projects/tfidf-fi/subjects.ttl')))
    subjectfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'yso-archaeology.ttl')
    result = runner.invoke(annif.cli.cli, ['loadvoc', 'tfidf-fi', subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('vocabs/yso-fi/subjects').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects').size() > 0
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').exists()
    assert testdatadir.join('vocabs/yso-fi/subjects.ttl').size() > 0


def test_loadvoc_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, [
            'loadvoc', 'dummy-fi', 'nonexistent_path'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Invalid value for 'SUBJECTFILE': " \
           "File 'nonexistent_path' does not exist." in failed_result.output


def test_train(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli, ['train', 'tfidf-fi', docfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/vectorizer').exists()
    assert testdatadir.join('projects/tfidf-fi/vectorizer').size() > 0
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').exists()
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').size() > 0


def test_train_multiple(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli,
                           ['train', 'tfidf-fi', docfile, docfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join('projects/tfidf-fi/vectorizer').exists()
    assert testdatadir.join('projects/tfidf-fi/vectorizer').size() > 0
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').exists()
    assert testdatadir.join('projects/tfidf-fi/tfidf-index').size() > 0


def test_train_cached(testdatadir):
    result = runner.invoke(annif.cli.cli,
                           ['train', '--cached', 'tfidf-fi'])
    assert result.exception
    assert result.exit_code == 1
    assert 'Training tfidf project from cached data not supported.' \
           in result.output


def test_train_cached_with_corpus(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli,
                           ['train', '--cached', 'tfidf-fi', docfile])
    assert result.exception
    assert result.exit_code == 2
    assert 'Corpus paths cannot be given when using --cached option.' \
           in result.output


def test_train_param_override_algo_notsupported():
    pytest.importorskip('annif.backend.vw_multi')
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(
        annif.cli.cli,
        ['train', 'vw-multi-fi', docfile,
         '--backend-param', 'vw_multi.algorithm=oaa'])
    assert result.exception
    assert result.exit_code == 1
    assert 'Algorithm overriding not supported.' in result.output


def test_train_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, [
            'train', 'dummy-fi', 'nonexistent_path'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Invalid value for '[PATHS]...': " \
           "Path 'nonexistent_path' does not exist." in failed_result.output


def test_train_no_path(caplog):
    logger = annif.logger
    logger.propagate = True
    result = runner.invoke(
        annif.cli.cli, [
            'train', 'dummy-fi'])
    assert not result.exception
    assert result.exit_code == 0
    assert 'Reading empty file' == caplog.records[0].message


def test_train_docslimit_zero():
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    failed_result = runner.invoke(
        annif.cli.cli, [
            'train', 'tfidf-fi', docfile, '--docs-limit', '0'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Not supported: Cannot train tfidf project with no documents"  \
        in failed_result.output


def test_learn(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli, ['learn', 'dummy-fi', docfile])
    assert not result.exception
    assert result.exit_code == 0


def test_learn_notsupported(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli, ['learn', 'tfidf-fi', docfile])
    assert result.exit_code != 0
    assert 'Learning not supported' in result.output


def test_learn_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, [
            'learn', 'dummy-fi', 'nonexistent_path'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Invalid value for '[PATHS]...': " \
           "Path 'nonexistent_path' does not exist." in failed_result.output


def test_suggest():
    result = runner.invoke(
        annif.cli.cli,
        ['suggest', 'dummy-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t1.0\n"
    assert result.exit_code == 0


def test_suggest_with_notations():
    result = runner.invoke(
        annif.cli.cli,
        ['suggest', '--backend-param', 'dummy.notation=42.42', 'dummy-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t42.42\t1.0\n"
    assert result.exit_code == 0


def test_suggest_nonexistent():
    result = runner.invoke(
        annif.cli.cli,
        ['suggest', TEMP_PROJECT],
        input='kissa')
    assert result.exception
    assert result.output == "No projects found with id '{}'.\n".format(
        TEMP_PROJECT)
    assert result.exit_code != 0


def test_suggest_param():
    result = runner.invoke(
        annif.cli.cli,
        ['suggest', '--backend-param', 'dummy.score=0.8', 'dummy-fi'],
        input='kissa')
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t0.8\n"
    assert result.exit_code == 0


def test_suggest_param_backend_nonexistent():
    result = runner.invoke(
        annif.cli.cli,
        ['suggest', '--backend-param', 'not_a_backend.score=0.8', 'dummy-fi'],
        input='kissa')
    assert result.exception
    assert 'The backend not_a_backend in CLI option ' + \
        '"-b not_a_backend.score=0.8" not matching the project backend ' + \
        'dummy.' in result.output
    assert result.exit_code != 0


def test_suggest_ensemble():
    result = runner.invoke(
        annif.cli.cli,
        ['suggest', 'ensemble'],
        input='the cat sat on the mat')
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t1.0\n"
    assert result.exit_code == 0


def test_index(tmpdir):
    tmpdir.join('doc1.txt').write('nothing special')

    result = runner.invoke(
        annif.cli.cli, ['index', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    assert tmpdir.join('doc1.annif').exists()
    assert tmpdir.join('doc1.annif').read_text(
        'utf-8') == "<http://example.org/dummy>\tdummy\t1.0\n"

    # make sure that preexisting subject files are not overwritten
    result = runner.invoke(
        annif.cli.cli, ['index', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0
    assert "Not overwriting" in result.output

    # check that the --force parameter forces overwriting
    result = runner.invoke(
        annif.cli.cli, ['index', 'dummy-fi', '--force', str(tmpdir)])
    assert tmpdir.join('doc1.annif').exists()
    assert "Not overwriting" not in result.output
    assert tmpdir.join('doc1.annif').read_text(
        'utf-8') == "<http://example.org/dummy>\tdummy\t1.0\n"


def test_index_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, [
            'index', 'dummy-fi', 'nonexistent_path'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Invalid value for 'DIRECTORY': " \
           "Directory 'nonexistent_path' does not exist." \
           in failed_result.output


def test_eval_label(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(annif.cli.cli, ['eval', 'dummy-en', str(tmpdir)])
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


def test_eval_uri(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write("<http://example.org/dummy>\tdummy\n")
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write("<http://example.org/none>\tnone\n")
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(annif.cli.cli, ['eval', 'dummy-en', str(tmpdir)])
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


def test_eval_param(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(
        annif.cli.cli, [
            'eval', '--backend-param', 'dummy.score=0.0', 'dummy-en',
            str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    # since zero scores were set with the parameter, there should be no hits
    # at all
    recall = re.search(r'Recall .*doc.*:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.0


def test_eval_resultsfile(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')
    resultfile = tmpdir.join('results.tsv')
    result = runner.invoke(
        annif.cli.cli, [
            'eval', '--results-file', str(resultfile), 'dummy-en',
            str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    # subject average should equal average of all subject scores in outputfile
    precision = float(
        re.search(r'Precision .*subj.*:\s+(\d.\d+)', result.output).group(1))
    recall = float(
        re.search(r'Recall .*subj.*:\s+(\d.\d+)', result.output).group(1))
    f_measure = float(
        re.search(r'F1 score .*subj.*:\s+(\d.\d+)', result.output).group(1))
    precision_numerator = 0
    recall_numerator = 0
    f_measure_numerator = 0
    denominator = 0
    with resultfile.open() as f:
        header = next(f)
        assert header.strip('\n') == '\t'.join(['URI',
                                                'Label',
                                                'Support',
                                                'True_positives',
                                                'False_positives',
                                                'False_negatives',
                                                'Precision',
                                                'Recall',
                                                'F1_score'])
        for line in f:
            assert line.strip() != ''
            parts = line.split('\t')
            if parts[1] == 'dummy':
                assert int(parts[2]) == 1
                assert int(parts[3]) == 1
                assert int(parts[4]) == 1
                assert int(parts[5]) == 0
            if parts[1] == 'none':
                assert int(parts[2]) == 1
                assert int(parts[3]) == 0
                assert int(parts[4]) == 0
                assert int(parts[5]) == 1
            precision_numerator += float(parts[6])
            recall_numerator += float(parts[7])
            f_measure_numerator += float(parts[8])
            denominator += 1
    assert precision_numerator / denominator == precision
    assert recall_numerator / denominator == recall
    assert f_measure_numerator / denominator == f_measure


def test_eval_badresultsfile(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')
    failed_result = runner.invoke(
        annif.cli.cli, [
            'eval', '--results-file', 'newdir/test_file.txt', 'dummy-en',
            str(tmpdir)])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert 'cannot open results-file for writing' in failed_result.output


def test_eval_docfile():
    docfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'documents.tsv')
    result = runner.invoke(annif.cli.cli, ['eval', 'dummy-fi', docfile])
    assert not result.exception
    assert result.exit_code == 0


def test_eval_empty_file(tmpdir):
    empty_file = tmpdir.ensure('empty.tsv')
    failed_result = runner.invoke(
        annif.cli.cli, [
            'eval', 'dummy-fi', str(empty_file)])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert 'cannot evaluate empty corpus' in failed_result.output


def test_eval_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, [
            'eval', 'dummy-fi', 'nonexistent_path'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Invalid value for '[PATHS]...': " \
           "Path 'nonexistent_path' does not exist." in failed_result.output


def test_eval_single_process(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(
        annif.cli.cli, ['eval', '--jobs', '1', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0


def test_eval_two_jobs(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    tmpdir.join('doc3.txt').write('doc3')

    result = runner.invoke(
        annif.cli.cli, ['eval', '--jobs', '2', 'dummy-en', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0


def test_optimize_dir(tmpdir):
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

    precision = re.search(r'Best\s+Precision .*?doc.*?:\s+(\d.\d+)',
                          result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search(r'Best\s+Recall .*?doc.*?:\s+(\d.\d+)', result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r'Best\s+F1 score .*?doc.*?:\s+(\d.\d+)',
                          result.output)
    assert float(f_measure.group(1)) == 0.5
    ndocs = re.search(r'Documents evaluated:\s+(\d)', result.output)
    assert int(ndocs.group(1)) == 2


def test_optimize_docfile(tmpdir):
    docfile = tmpdir.join('documents.tsv')
    docfile.write("""Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>""")

    result = runner.invoke(
        annif.cli.cli, [
            'optimize', 'dummy-fi', str(docfile)])
    assert not result.exception
    assert result.exit_code == 0


def test_optimize_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, [
            'optimize', 'dummy-fi', 'nonexistent_path'])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Invalid value for '[PATHS]...': " \
           "Path 'nonexistent_path' does not exist." in failed_result.output


def test_hyperopt_ensemble(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')

    result = runner.invoke(
        annif.cli.cli, [
            'hyperopt', 'ensemble', str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    assert re.search(
        r'sources=dummy-en:0.\d+,dummydummy:0.\d+',
        result.output) is not None


def test_hyperopt_ensemble_resultsfile(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')
    resultfile = tmpdir.join('results.tsv')

    result = runner.invoke(
        annif.cli.cli, [
            'hyperopt', '--results-file', str(resultfile), 'ensemble',
            str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    with resultfile.open() as f:
        header = next(f)
        assert header.strip('\n') == '\t'.join(['trial',
                                                'value',
                                                'dummy-en',
                                                'dummydummy'])
        for idx, line in enumerate(f):
            assert line.strip() != ''
            parts = line.split('\t')
            assert len(parts) == 4
            assert int(parts[0]) == idx


def test_hyperopt_not_supported(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('dummy')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('none')

    failed_result = runner.invoke(
        annif.cli.cli, [
            'hyperopt', 'tfidf-en', str(tmpdir)])
    assert failed_result.exception
    assert failed_result.exit_code != 0

    assert 'Hyperparameter optimization not supported' in failed_result.output


def test_version_option():
    result = runner.invoke(
        annif.cli.cli, ['--version'])
    assert not result.exception
    assert result.exit_code == 0
    version = pkg_resources.require('annif')[0].version
    assert result.output.strip() == version.strip()
