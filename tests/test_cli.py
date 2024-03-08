"""Unit test module for Annif CLI commands"""

import contextlib
import importlib
import io
import json
import os.path
import random
import re
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from unittest import mock

import huggingface_hub
from click.shell_completion import ShellComplete
from click.testing import CliRunner
from huggingface_hub.utils import HFValidationError

import annif.cli
import annif.cli_util
import annif.parallel

runner = CliRunner(env={"ANNIF_CONFIG": "annif.default_config.TestingConfig"})

# Generate a random project name to use in tests
TEMP_PROJECT = "".join(random.choice("abcdefghiklmnopqrstuvwxyz") for _ in range(8))
PROJECTS_CONFIG_PATH = "tests/projects_for_config_path_option.cfg"


@mock.patch.dict(os.environ, clear=True)
def test_tensorflow_loglevel():
    tf_env = "TF_CPP_MIN_LOG_LEVEL"

    runner.invoke(annif.cli.cli, ["list-projects", "-v", "DEBUG"])
    assert os.environ[tf_env] == "0"  # Show INFO, WARNING and ERROR messages by TF
    os.environ.pop(tf_env)
    runner.invoke(annif.cli.cli, ["list-projects"])  # INFO level by default
    assert os.environ[tf_env] == "1"  # Show WARNING and ERROR messages by TF
    os.environ.pop(tf_env)
    runner.invoke(annif.cli.cli, ["list-projects", "-v", "WARN"])
    assert os.environ[tf_env] == "1"  # Show WARNING and ERROR messages by TF
    os.environ.pop(tf_env)
    runner.invoke(annif.cli.cli, ["list-projects", "-v", "ERROR"])
    assert os.environ[tf_env] == "2"  # Show ERROR messages by TF
    os.environ.pop(tf_env)
    runner.invoke(annif.cli.cli, ["list-projects", "-v", "CRITICAL"])
    assert os.environ[tf_env] == "3"  # Show no messages by TF
    os.environ.pop(tf_env)


def test_list_projects():
    result = runner.invoke(annif.cli.cli, ["list-projects"])
    assert not result.exception
    assert result.exit_code == 0
    # public project should be visible
    assert "dummy-fi" in result.output
    # hidden project should be visible
    assert "dummy-en" in result.output
    # private project should be visible
    assert "dummy-private" in result.output
    # project with no access setting should be visible
    assert "ensemble" in result.output


def test_list_projects_bad_arguments():
    # The listprojects function does not accept any arguments, it should fail
    # if such are provided.
    assert runner.invoke(annif.cli.cli, ["list-projects", "moi"]).exit_code != 0
    assert (
        runner.invoke(annif.cli.run_list_projects, ["moi", "--debug", "y"]).exit_code
        != 0
    )


def test_list_projects_config_path_option():
    result = runner.invoke(
        annif.cli.cli, ["list-projects", "--projects", PROJECTS_CONFIG_PATH]
    )
    assert not result.exception
    assert result.exit_code == 0
    assert "dummy_for_projects_option" in result.output
    assert "dummy-fi" not in result.output
    assert "dummy-en" not in result.output


def test_list_projects_config_path_option_nonexistent():
    failed_result = runner.invoke(
        annif.cli.cli, ["list-projects", "--projects", "nonexistent.cfg"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Error: Invalid value for '-p' / '--projects': "
        "Path 'nonexistent.cfg' does not exist." in failed_result.output
    )


def test_show_project():
    result = runner.invoke(annif.cli.cli, ["show-project", "dummy-en"])
    assert not result.exception

    project_id = re.search(r"Project ID:\s+(.+)", result.output)
    assert project_id.group(1) == "dummy-en"
    project_name = re.search(r"Project Name:\s+(.+)", result.output)
    assert project_name.group(1) == "Dummy English"
    project_lang = re.search(r"Language:\s+(.+)", result.output)
    assert project_lang.group(1) == "en"
    access = re.search(r"Access:\s+(.+)", result.output)
    assert access.group(1) == "hidden"
    access = re.search(r"Backend:\s+(.+)", result.output)
    assert access.group(1) == "dummy"
    is_trained = re.search(r"Trained:\s+(.+)", result.output)
    assert is_trained.group(1) == "True"
    modification_time = re.search(r"Modification time:\s+(.+)", result.output)
    assert modification_time.group(1) == "-"


def test_show_project_modification_time(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "tfidf-fi")
    fpath = os.path.join(str(dirpath), "test_show_project_datafile")
    os.makedirs(dirpath)
    open(fpath, "a").close()

    result = runner.invoke(annif.cli.cli, ["show-project", "tfidf-fi"])
    assert not result.exception
    modification_time = re.search(r"Modification time:\s+(.+)", result.output)
    modification_time_obj = datetime.strptime(
        modification_time.group(1), "%Y-%m-%d %H:%M:%S"
    )
    assert datetime.now() - modification_time_obj < timedelta(1)


def test_show_project_nonexistent():
    assert runner.invoke(annif.cli.cli, ["show-project", TEMP_PROJECT]).exit_code != 0
    # Test should not fail even if the user queries for a non-existent project.
    failed_result = runner.invoke(annif.cli.cli, ["show-project", "nonexistent"])
    assert failed_result.exception


def test_clear_project(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "test_clear_project_datafile")
    os.makedirs(dirpath)
    open(fpath, "a").close()

    assert runner.invoke(annif.cli.cli, ["clear", "dummy-fi"]).exit_code == 0
    assert not os.path.isdir(dirpath)


def test_clear_project_nonexistent_data(testdatadir, caplog):
    logger = annif.logger
    logger.propagate = True
    result = runner.invoke(annif.cli.cli, ["clear", "dummy-fi"])
    assert not result.exception
    assert result.exit_code == 0
    assert len(caplog.records) == 1
    expected_msg = "No model data to remove for project dummy-fi."
    assert expected_msg == caplog.records[0].message


def test_list_vocabs_before_load(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(str(testdatadir.join("vocabs/yso/")))
    result = runner.invoke(annif.cli.cli, ["list-vocabs"])
    assert not result.exception
    assert result.exit_code == 0
    assert re.search(r"^yso\s+-\s+-\s+False", result.output, re.MULTILINE)


def test_load_vocab_csv(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.csv")))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.ttl")))
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "subjects.csv"
    )
    result = runner.invoke(annif.cli.cli, ["load-vocab", "yso", subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.ttl").exists()
    assert testdatadir.join("vocabs/yso/subjects.ttl").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").exists()
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").size() > 0


def test_load_vocab_tsv(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.csv")))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.ttl")))
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "subjects.tsv"
    )
    result = runner.invoke(
        annif.cli.cli, ["load-vocab", "--language", "fi", "yso", subjectfile]
    )
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.ttl").exists()
    assert testdatadir.join("vocabs/yso/subjects.ttl").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").exists()
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").size() > 0


def test_load_vocab_tsv_no_lang(testdatadir):
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "subjects.tsv"
    )
    failed_result = runner.invoke(annif.cli.cli, ["load-vocab", "yso", subjectfile])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Please use --language option to set the language "
        "of a TSV vocabulary." in failed_result.output
    )


def test_load_vocab_tsv_with_bom(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.csv")))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.ttl")))
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "subjects-bom.tsv"
    )
    result = runner.invoke(
        annif.cli.cli, ["load-vocab", "--language", "fi", "yso", subjectfile]
    )
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.ttl").exists()
    assert testdatadir.join("vocabs/yso/subjects.ttl").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").exists()
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").size() > 0


def test_load_vocab_rdf(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.csv")))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.ttl")))
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "yso-archaeology.rdf"
    )
    result = runner.invoke(annif.cli.cli, ["load-vocab", "yso", subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.ttl").exists()
    assert testdatadir.join("vocabs/yso/subjects.ttl").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").exists()
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").size() > 0


def test_load_vocab_ttl(testdatadir):
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.csv")))
    with contextlib.suppress(FileNotFoundError):
        os.remove(str(testdatadir.join("vocabs/yso/subjects.ttl")))
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "yso-archaeology.ttl"
    )
    result = runner.invoke(annif.cli.cli, ["load-vocab", "yso", subjectfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.ttl").exists()
    assert testdatadir.join("vocabs/yso/subjects.ttl").size() > 0
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").exists()
    assert testdatadir.join("vocabs/yso/subjects.dump.gz").size() > 0


def test_load_vocab_nonexistent_vocab():
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "yso-archaeology.ttl"
    )
    failed_result = runner.invoke(
        annif.cli.cli, ["load-vocab", "notfound", subjectfile]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "No vocabularies found with the id 'notfound'." in failed_result.output


def test_load_vocab_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, ["load-vocab", "dummy", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for 'SUBJECTFILE': "
        "File 'nonexistent_path' does not exist." in failed_result.output
    )


def test_list_vocabs_after_load():
    result = runner.invoke(annif.cli.cli, ["list-vocabs"])
    assert not result.exception
    assert result.exit_code == 0
    assert re.search(r"^dummy\s+en,fi\s+2\s+True", result.output, re.MULTILINE)
    assert re.search(r"^yso\s+en,fi,sv\s+130\s+True", result.output, re.MULTILINE)


def test_train(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    result = runner.invoke(annif.cli.cli, ["train", "tfidf-fi", docfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("projects/tfidf-fi/vectorizer").exists()
    assert testdatadir.join("projects/tfidf-fi/vectorizer").size() > 0
    assert testdatadir.join("projects/tfidf-fi/tfidf-index").exists()
    assert testdatadir.join("projects/tfidf-fi/tfidf-index").size() > 0


def test_train_multiple(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    result = runner.invoke(annif.cli.cli, ["train", "tfidf-fi", docfile, docfile])
    assert not result.exception
    assert result.exit_code == 0
    assert testdatadir.join("projects/tfidf-fi/vectorizer").exists()
    assert testdatadir.join("projects/tfidf-fi/vectorizer").size() > 0
    assert testdatadir.join("projects/tfidf-fi/tfidf-index").exists()
    assert testdatadir.join("projects/tfidf-fi/tfidf-index").size() > 0


def test_train_cached(testdatadir):
    result = runner.invoke(annif.cli.cli, ["train", "--cached", "tfidf-fi"])
    assert result.exception
    assert result.exit_code == 1
    assert "Training tfidf project from cached data not supported." in result.output


def test_train_cached_with_corpus(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    result = runner.invoke(annif.cli.cli, ["train", "--cached", "tfidf-fi", docfile])
    assert result.exception
    assert result.exit_code == 2
    assert "Corpus paths cannot be given when using --cached option." in result.output


def test_train_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, ["train", "dummy-fi", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for '[PATHS]...': "
        "Path 'nonexistent_path' does not exist." in failed_result.output
    )


def test_train_no_path(caplog):
    logger = annif.logger
    logger.propagate = True
    result = runner.invoke(annif.cli.cli, ["train", "dummy-fi"])
    assert not result.exception
    assert result.exit_code == 0
    assert "Reading empty file" == caplog.records[0].message


def test_train_docslimit_zero():
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    failed_result = runner.invoke(
        annif.cli.cli, ["train", "tfidf-fi", docfile, "--docs-limit", "0"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Not supported: Cannot train tfidf project with no documents"
        in failed_result.output
    )


def test_learn(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    result = runner.invoke(annif.cli.cli, ["learn", "dummy-fi", docfile])
    assert not result.exception
    assert result.exit_code == 0


def test_learn_notsupported(testdatadir):
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    result = runner.invoke(annif.cli.cli, ["learn", "tfidf-fi", docfile])
    assert result.exit_code != 0
    assert "Learning not supported" in result.output


def test_learn_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, ["learn", "dummy-fi", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for '[PATHS]...': "
        "Path 'nonexistent_path' does not exist." in failed_result.output
    )


def test_suggest():
    result = runner.invoke(annif.cli.cli, ["suggest", "dummy-fi"], input="kissa")
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy-fi\t1.0000\n"
    assert result.exit_code == 0


def test_suggest_with_language_override():
    result = runner.invoke(
        annif.cli.cli, ["suggest", "--language", "en", "dummy-fi"], input="kissa"
    )
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t1.0000\n"
    assert result.exit_code == 0


def test_suggest_with_language_override_bad_value():
    failed_result = runner.invoke(
        annif.cli.cli, ["suggest", "--language", "xx", "dummy-fi"], input="kissa"
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert 'language "xx" not supported by vocabulary' in failed_result.output


def test_suggest_with_different_vocab_language():
    # project language is English - input should be in English
    # vocab language is Finnish - subject labels should be in Finnish
    result = runner.invoke(
        annif.cli.cli, ["suggest", "dummy-vocablang"], input="the cat sat on the mat"
    )
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy-fi\t1.0000\n"
    assert result.exit_code == 0


def test_suggest_with_notations():
    result = runner.invoke(
        annif.cli.cli,
        ["suggest", "--backend-param", "dummy.uri=http://example.org/none", "dummy-fi"],
        input="kissa",
    )
    assert not result.exception
    assert result.output == "<http://example.org/none>\tnone-fi\t42.42\t1.0000\n"
    assert result.exit_code == 0


def test_suggest_nonexistent():
    result = runner.invoke(annif.cli.cli, ["suggest", TEMP_PROJECT], input="kissa")
    assert result.exception
    assert result.output == "No projects found with id '{}'.\n".format(TEMP_PROJECT)
    assert result.exit_code != 0


def test_suggest_param():
    result = runner.invoke(
        annif.cli.cli,
        ["suggest", "--backend-param", "dummy.score=0.8", "dummy-fi"],
        input="kissa",
    )
    assert not result.exception
    assert result.output.startswith("<http://example.org/dummy>\tdummy-fi\t0.8")
    assert result.exit_code == 0


def test_suggest_param_backend_nonexistent():
    result = runner.invoke(
        annif.cli.cli,
        ["suggest", "--backend-param", "not_a_backend.score=0.8", "dummy-fi"],
        input="kissa",
    )
    assert result.exception
    assert (
        "The backend not_a_backend in CLI option "
        + '"-b not_a_backend.score=0.8" not matching the project backend '
        + "dummy."
        in result.output
    )
    assert result.exit_code != 0


def test_suggest_ensemble():
    result = runner.invoke(
        annif.cli.cli, ["suggest", "ensemble"], input="the cat sat on the mat"
    )
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy\t1.0000\n"
    assert result.exit_code == 0


def test_suggest_file(tmpdir):
    docfile = tmpdir.join("doc.txt")
    docfile.write("nothing special")

    result = runner.invoke(annif.cli.cli, ["suggest", "dummy-fi", str(docfile)])

    assert not result.exception
    assert f"Suggestions for {docfile}" in result.output
    assert "<http://example.org/dummy>\tdummy-fi\t1.0000\n" in result.output
    assert result.exit_code == 0


def test_suggest_two_files(tmpdir):
    docfile1 = tmpdir.join("doc-1.txt")
    docfile1.write("nothing special")
    docfile2 = tmpdir.join("doc-2.txt")
    docfile2.write("again nothing special")

    result = runner.invoke(
        annif.cli.cli, ["suggest", "dummy-fi", str(docfile1), str(docfile2)]
    )

    assert not result.exception
    assert f"Suggestions for {docfile1}" in result.output
    assert f"Suggestions for {docfile2}" in result.output
    assert result.output.count("<http://example.org/dummy>\tdummy-fi\t1.0000\n") == 2
    assert result.exit_code == 0


def test_suggest_two_files_docs_limit(tmpdir):
    docfile1 = tmpdir.join("doc-1.txt")
    docfile1.write("nothing special")
    docfile2 = tmpdir.join("doc-2.txt")
    docfile2.write("again nothing special")

    result = runner.invoke(
        annif.cli.cli,
        ["suggest", "dummy-fi", str(docfile1), str(docfile2), "--docs-limit", "1"],
    )

    assert not result.exception
    assert f"Suggestions for {docfile1}" in result.output
    assert f"Suggestions for {docfile2}" not in result.output
    assert result.output.count("<http://example.org/dummy>\tdummy-fi\t1.0000\n") == 1
    assert result.exit_code == 0


def test_suggest_file_and_stdin(tmpdir):
    docfile1 = tmpdir.join("doc-1.txt")
    docfile1.write("nothing special")

    result = runner.invoke(
        annif.cli.cli, ["suggest", "dummy-fi", str(docfile1), "-"], input="kissa"
    )

    assert not result.exception
    assert f"Suggestions for {docfile1}" in result.output
    assert "Suggestions for -" in result.output
    assert result.output.count("<http://example.org/dummy>\tdummy-fi\t1.0000\n") == 2
    assert result.exit_code == 0


def test_suggest_file_nonexistent():
    failed_result = runner.invoke(
        annif.cli.cli, ["suggest", "dummy-fi", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for '[PATHS]...': "
        "File 'nonexistent_path' does not exist." in failed_result.output
    )


def test_suggest_dash_path():
    result = runner.invoke(
        annif.cli.cli, ["suggest", "dummy-fi", "-"], input="the cat sat on the mat"
    )
    assert not result.exception
    assert result.output == "<http://example.org/dummy>\tdummy-fi\t1.0000\n"
    assert result.exit_code == 0


def test_index(tmpdir):
    tmpdir.join("doc1.txt").write("nothing special")
    # Existing subject files should not have an effect
    tmpdir.join("doc1.tsv").write("<http://example.org/dummy>\tdummy")
    tmpdir.join("doc1.key").write("<http://example.org/dummy>\tdummy")

    result = runner.invoke(annif.cli.cli, ["index", "dummy-en", str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    assert tmpdir.join("doc1.annif").exists()
    assert (
        tmpdir.join("doc1.annif").read_text("utf-8")
        == "<http://example.org/dummy>\tdummy\t1.0000\n"
    )

    # make sure that preexisting subject files are not overwritten
    result = runner.invoke(annif.cli.cli, ["index", "dummy-en", str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0
    assert "Not overwriting" in result.output

    # check that the --force parameter forces overwriting
    result = runner.invoke(annif.cli.cli, ["index", "dummy-fi", "--force", str(tmpdir)])
    assert tmpdir.join("doc1.annif").exists()
    assert "Not overwriting" not in result.output
    assert (
        tmpdir.join("doc1.annif").read_text("utf-8")
        == "<http://example.org/dummy>\tdummy-fi\t1.0000\n"
    )


def test_index_with_language_override(tmpdir):
    tmpdir.join("doc1.txt").write("nothing special")

    result = runner.invoke(
        annif.cli.cli, ["index", "--language", "fi", "dummy-en", str(tmpdir)]
    )
    assert not result.exception
    assert result.exit_code == 0

    assert tmpdir.join("doc1.annif").exists()
    assert (
        tmpdir.join("doc1.annif").read_text("utf-8")
        == "<http://example.org/dummy>\tdummy-fi\t1.0000\n"
    )


def test_index_with_language_override_bad_value(tmpdir):
    tmpdir.join("doc1.txt").write("nothing special")

    failed_result = runner.invoke(
        annif.cli.cli, ["index", "--language", "xx", "dummy-en", str(tmpdir)]
    )

    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert 'language "xx" not supported by vocabulary' in failed_result.output


def test_index_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, ["index", "dummy-fi", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for 'DIRECTORY': "
        "Directory 'nonexistent_path' does not exist." in failed_result.output
    )


def test_eval_label(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")

    result = runner.invoke(annif.cli.cli, ["eval", "dummy-en", str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search(r"Precision .*doc.*:\s+(\d.\d+)", result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search(r"Recall .*doc.*:\s+(\d.\d+)", result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r"F1 score .*doc.*:\s+(\d.\d+)", result.output)
    assert float(f_measure.group(1)) == 0.5
    precision1 = re.search(r"Precision@1:\s+(\d.\d+)", result.output)
    assert float(precision1.group(1)) == 0.5
    precision3 = re.search(r"Precision@3:\s+(\d.\d+)", result.output)
    assert float(precision3.group(1)) == 0.5
    precision5 = re.search(r"Precision@5:\s+(\d.\d+)", result.output)
    assert float(precision5.group(1)) == 0.5
    true_positives = re.search(r"True positives:\s+(\d+)", result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search(r"False positives:\s+(\d+)", result.output)
    assert int(false_positives.group(1)) == 1
    false_negatives = re.search(r"False negatives:\s+(\d+)", result.output)
    assert int(false_negatives.group(1)) == 1
    ndocs = re.search(r"Documents evaluated:\s+(\d+)", result.output)
    assert int(ndocs.group(1)) == 2


def test_eval_uri(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("<http://example.org/dummy>\tdummy\n")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("<http://example.org/none>\tnone\n")
    tmpdir.join("doc3.txt").write("doc3")

    result = runner.invoke(annif.cli.cli, ["eval", "dummy-en", str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search(r"Precision .*doc.*:\s+(\d.\d+)", result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search(r"Recall .*doc.*:\s+(\d.\d+)", result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r"F1 score .*doc.*:\s+(\d.\d+)", result.output)
    assert float(f_measure.group(1)) == 0.5
    precision1 = re.search(r"Precision@1:\s+(\d.\d+)", result.output)
    assert float(precision1.group(1)) == 0.5
    precision3 = re.search(r"Precision@3:\s+(\d.\d+)", result.output)
    assert float(precision3.group(1)) == 0.5
    precision5 = re.search(r"Precision@5:\s+(\d.\d+)", result.output)
    assert float(precision5.group(1)) == 0.5
    true_positives = re.search(r"True positives:\s+(\d+)", result.output)
    assert int(true_positives.group(1)) == 1
    false_positives = re.search(r"False positives:\s+(\d+)", result.output)
    assert int(false_positives.group(1)) == 1
    false_negatives = re.search(r"False negatives:\s+(\d+)", result.output)
    assert int(false_negatives.group(1)) == 1
    ndocs = re.search(r"Documents evaluated:\s+(\d+)", result.output)
    assert int(ndocs.group(1)) == 2


def test_eval_param(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")

    result = runner.invoke(
        annif.cli.cli,
        ["eval", "--backend-param", "dummy.score=0.0", "dummy-en", str(tmpdir)],
    )
    assert not result.exception
    assert result.exit_code == 0

    # since zero scores were set with the parameter, there should be no hits
    # at all
    recall = re.search(r"Recall .*doc.*:\s+(\d.\d+)", result.output)
    assert float(recall.group(1)) == 0.0


def test_eval_metric(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")
    result = runner.invoke(
        annif.cli.cli,
        ["eval", "--metric", "F1@5", "-m", "NDCG", "dummy-en", str(tmpdir)],
    )
    assert not result.exception
    assert result.exit_code == 0

    f1 = re.search(r"F1@5\s*:\s+(\d.\d+)", result.output)
    assert float(f1.group(1)) > 0.0
    ndcg = re.search(r"NDCG\s*:\s+(\d.\d+)", result.output)
    assert float(ndcg.group(1)) > 0.0
    # check that we only have 2 metrics + "Documents evaluated"
    assert len(result.output.strip().split("\n")) == 3


def test_eval_metricsfile(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")
    metricsfile = tmpdir.join("metrics.json")
    result = runner.invoke(
        annif.cli.cli,
        ["eval", "--metrics-file", str(metricsfile), "dummy-en", str(tmpdir)],
    )
    assert not result.exception
    assert result.exit_code == 0

    metrics = json.load(metricsfile)
    assert "F1@5" in metrics
    assert metrics["F1@5"] > 0.0
    assert "NDCG" in metrics
    assert metrics["NDCG"] > 0.0
    assert "Precision_doc_avg" in metrics
    assert metrics["Precision_doc_avg"] > 0.0


def test_eval_resultsfile(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")
    resultfile = tmpdir.join("results.tsv")
    result = runner.invoke(
        annif.cli.cli,
        ["eval", "--results-file", str(resultfile), "dummy-en", str(tmpdir)],
    )
    assert not result.exception
    assert result.exit_code == 0

    # subject average should equal average of all subject scores in outputfile
    precision = float(
        re.search(r"Precision .*subj.*:\s+(\d.\d+)", result.output).group(1)
    )
    recall = float(re.search(r"Recall .*subj.*:\s+(\d.\d+)", result.output).group(1))
    f_measure = float(
        re.search(r"F1 score .*subj.*:\s+(\d.\d+)", result.output).group(1)
    )
    precision_numerator = 0
    recall_numerator = 0
    f_measure_numerator = 0
    denominator = 0
    with resultfile.open() as f:
        header = next(f)
        assert header.strip("\n") == "\t".join(
            [
                "URI",
                "Label",
                "Support",
                "True_positives",
                "False_positives",
                "False_negatives",
                "Precision",
                "Recall",
                "F1_score",
            ]
        )
        for line in f:
            assert line.strip() != ""
            parts = line.split("\t")
            if parts[1] == "dummy":
                assert int(parts[2]) == 1
                assert int(parts[3]) == 1
                assert int(parts[4]) == 1
                assert int(parts[5]) == 0
            if parts[1] == "none":
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
    assert round(f_measure_numerator / denominator, 4) == f_measure


def test_eval_badresultsfile(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")
    failed_result = runner.invoke(
        annif.cli.cli,
        ["eval", "--results-file", "newdir/test_file.txt", "dummy-en", str(tmpdir)],
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "cannot open results-file for writing" in failed_result.output


def test_eval_docfile():
    docfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "documents.tsv"
    )
    result = runner.invoke(annif.cli.cli, ["eval", "dummy-fi", docfile])
    assert not result.exception
    assert result.exit_code == 0


def test_eval_empty_file(tmpdir):
    empty_file = tmpdir.ensure("empty.tsv")
    failed_result = runner.invoke(annif.cli.cli, ["eval", "dummy-fi", str(empty_file)])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "cannot evaluate empty corpus" in failed_result.output


def test_eval_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, ["eval", "dummy-fi", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for '[PATHS]...': "
        "Path 'nonexistent_path' does not exist." in failed_result.output
    )


def test_eval_single_process(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")

    result = runner.invoke(
        annif.cli.cli, ["eval", "--jobs", "1", "dummy-en", str(tmpdir)]
    )
    assert not result.exception
    assert result.exit_code == 0


def test_eval_two_jobs(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")

    result = runner.invoke(
        annif.cli.cli, ["eval", "--jobs", "2", "dummy-en", str(tmpdir)]
    )
    assert not result.exception
    assert result.exit_code == 0


def test_eval_two_jobs_spawn(tmpdir, monkeypatch):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")

    # use spawn method for starting multiprocessing worker processes
    monkeypatch.setattr(annif.parallel, "MP_START_METHOD", "spawn")
    result = runner.invoke(
        annif.cli.cli, ["eval", "--jobs", "2", "dummy-en", str(tmpdir)]
    )
    assert not result.exception
    assert result.exit_code == 0


def test_optimize_dir(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    tmpdir.join("doc3.txt").write("doc3")

    result = runner.invoke(annif.cli.cli, ["optimize", "dummy-en", str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    precision = re.search(r"Best\s+Precision .*?doc.*?:\s+(\d.\d+)", result.output)
    assert float(precision.group(1)) == 0.5
    recall = re.search(r"Best\s+Recall .*?doc.*?:\s+(\d.\d+)", result.output)
    assert float(recall.group(1)) == 0.5
    f_measure = re.search(r"Best\s+F1 score .*?doc.*?:\s+(\d.\d+)", result.output)
    assert float(f_measure.group(1)) == 0.5
    ndocs = re.search(r"Documents evaluated:\s+(\d)", result.output)
    assert int(ndocs.group(1)) == 2


def test_optimize_docfile(tmpdir):
    docfile = tmpdir.join("documents.tsv")
    docfile.write(
        """Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    )

    result = runner.invoke(annif.cli.cli, ["optimize", "dummy-fi", str(docfile)])
    assert not result.exception
    assert result.exit_code == 0


def test_optimize_nonexistent_path():
    failed_result = runner.invoke(
        annif.cli.cli, ["optimize", "dummy-fi", "nonexistent_path"]
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert (
        "Invalid value for '[PATHS]...': "
        "Path 'nonexistent_path' does not exist." in failed_result.output
    )


def test_hyperopt_ensemble(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")

    result = runner.invoke(annif.cli.cli, ["hyperopt", "ensemble", str(tmpdir)])
    assert not result.exception
    assert result.exit_code == 0

    assert (
        re.search(r"sources=dummy-en:0.\d+,dummy-private:0.\d+", result.output)
        is not None
    )


def test_hyperopt_ensemble_resultsfile(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")
    resultfile = tmpdir.join("results.tsv")

    result = runner.invoke(
        annif.cli.cli,
        ["hyperopt", "--results-file", str(resultfile), "ensemble", str(tmpdir)],
    )
    assert not result.exception
    assert result.exit_code == 0

    with resultfile.open() as f:
        header = next(f)
        assert header.strip("\n") == "\t".join(
            ["trial", "value", "dummy-en", "dummy-private"]
        )
        for idx, line in enumerate(f):
            assert line.strip() != ""
            parts = line.split("\t")
            assert len(parts) == 4
            assert int(parts[0]) == idx


def test_hyperopt_not_supported(tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.key").write("dummy")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.key").write("none")

    failed_result = runner.invoke(annif.cli.cli, ["hyperopt", "tfidf-en", str(tmpdir)])
    assert failed_result.exception
    assert failed_result.exit_code != 0

    assert "Hyperparameter optimization not supported" in failed_result.output


def test_version_option():
    result = runner.invoke(annif.cli.cli, ["--version"])
    assert not result.exception
    assert result.exit_code == 0
    version = importlib.metadata.version("annif")
    assert result.output.strip() == version.strip()


def test_run():
    result = runner.invoke(annif.cli.cli, ["run", "--help"])
    assert not result.exception
    assert result.exit_code == 0
    assert "Run a local development server." in result.output


def test_routes_with_flask_app():
    # When using plain Flask only the static endpoint exists
    result = runner.invoke(annif.cli.cli, ["routes"])
    assert re.search(r"static\s+GET\s+\/static\/\<path:filename\>", result.output)
    assert not re.search(r"app.home\s+GET\s+\/", result.output)


def test_routes_with_connexion_app():
    # When using Connexion all endpoints exist
    result = os.popen("python annif/cli.py routes").read()
    assert re.search(r"static\s+GET\s+\/static\/<path:filename>", result)
    assert re.search(r"app.home\s+GET\s+\/", result)


@mock.patch("huggingface_hub.HfApi.upload_file")
def test_upload(upload_file):
    result = runner.invoke(annif.cli.cli, ["upload", "dummy-fi", "dummy-repo"])
    assert not result.exception
    assert huggingface_hub.HfApi.upload_file.call_count == 3
    assert (
        mock.call(
            path_or_fileobj=mock.ANY,  # io.BufferedRandom object
            path_in_repo="data/vocabs/dummy.zip",
            repo_id="dummy-repo",
            token=None,
            commit_message="Upload project(s) dummy-fi with Annif",
        )
        in huggingface_hub.HfApi.upload_file.call_args_list
    )
    assert (
        mock.call(
            path_or_fileobj=mock.ANY,  # io.BufferedRandom object
            path_in_repo="data/projects/dummy-fi.zip",
            repo_id="dummy-repo",
            token=None,
            commit_message="Upload project(s) dummy-fi with Annif",
        )
        in huggingface_hub.HfApi.upload_file.call_args_list
    )
    assert (
        mock.call(
            path_or_fileobj=mock.ANY,  # io.BytesIO object
            path_in_repo="dummy-fi.cfg",
            repo_id="dummy-repo",
            token=None,
            commit_message="Upload project(s) dummy-fi with Annif",
        )
        in huggingface_hub.HfApi.upload_file.call_args_list
    )


@mock.patch("huggingface_hub.HfApi.upload_file")
def test_upload_many(upload_file):
    result = runner.invoke(annif.cli.cli, ["upload", "dummy-*", "dummy-repo"])
    assert not result.exception
    assert huggingface_hub.HfApi.upload_file.call_count == 11


def test_upload_nonexistent_repo():
    failed_result = runner.invoke(annif.cli.cli, ["upload", "dummy-fi", "nonexistent"])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Repository Not Found for url:" in failed_result.output


def test_archive_dir(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    os.makedirs(dirpath, exist_ok=True)
    open(os.path.join(str(dirpath), "foo.txt"), "a").close()
    open(os.path.join(str(dirpath), "-train.txt"), "a").close()

    fobj = annif.cli_util._archive_dir(dirpath)
    assert isinstance(fobj, io.BufferedRandom)

    with zipfile.ZipFile(fobj, mode="r") as zfile:
        archived_files = zfile.namelist()
    assert len(archived_files) == 1
    assert os.path.split(archived_files[0])[1] == "foo.txt"


def test_get_project_config(app_project):
    result = annif.cli_util._get_project_config(app_project)
    assert isinstance(result, io.BytesIO)
    string_result = result.read().decode("UTF-8")
    assert "[dummy-en]" in string_result


def hf_hub_download_mock_side_effect(filename, repo_id, token, revision):
    return "tests/huggingface-cache/" + filename  # Mocks the downloaded file paths


@mock.patch(
    "huggingface_hub.list_repo_files",
    return_value=[  # Mocks the filenames in repo
        "projects/dummy-fi.zip",
        "vocabs/dummy.zip",
        "dummy-fi.cfg",
        "projects/dummy-en.zip",
        "vocabs/dummy.zip",
        "dummy-.cfg",
    ],
)
@mock.patch(
    "huggingface_hub.hf_hub_download",
    side_effect=hf_hub_download_mock_side_effect,
)
@mock.patch("annif.cli_util.copy_project_config")
def test_download_dummy_fi(
    copy_project_config, hf_hub_download, list_repo_files, testdatadir
):
    result = runner.invoke(
        annif.cli.cli,
        [
            "download",
            "dummy-fi",
            "mock-repo",
        ],
    )
    assert not result.exception
    assert list_repo_files.called
    assert hf_hub_download.called
    assert hf_hub_download.call_args_list == [
        mock.call(
            repo_id="mock-repo",
            filename="projects/dummy-fi.zip",
            token=None,
            revision=None,
        ),
        mock.call(
            repo_id="mock-repo",
            filename="dummy-fi.cfg",
            token=None,
            revision=None,
        ),
        mock.call(
            repo_id="mock-repo",
            filename="vocabs/dummy.zip",
            token=None,
            revision=None,
        ),
    ]
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    assert os.path.exists(fpath)
    assert copy_project_config.call_args_list == [
        mock.call("tests/huggingface-cache/dummy-fi.cfg", False)
    ]


@mock.patch(
    "huggingface_hub.list_repo_files",
    return_value=[  # Mock filenames in repo
        "projects/dummy-fi.zip",
        "vocabs/dummy.zip",
        "dummy-fi.cfg",
        "projects/dummy-en.zip",
        "vocabs/dummy.zip",
        "dummy-.cfg",
    ],
)
@mock.patch(
    "huggingface_hub.hf_hub_download",
    side_effect=hf_hub_download_mock_side_effect,
)
@mock.patch("annif.cli_util.copy_project_config")
def test_download_dummy_fi_and_en(
    copy_project_config, hf_hub_download, list_repo_files, testdatadir
):
    result = runner.invoke(
        annif.cli.cli,
        [
            "download",
            "dummy-??",
            "mock-repo",
        ],
    )
    assert not result.exception
    assert list_repo_files.called
    assert hf_hub_download.called
    assert hf_hub_download.call_args_list == [
        mock.call(
            repo_id="mock-repo",
            filename="projects/dummy-fi.zip",
            token=None,
            revision=None,
        ),
        mock.call(
            repo_id="mock-repo",
            filename="dummy-fi.cfg",
            token=None,
            revision=None,
        ),
        mock.call(
            repo_id="mock-repo",
            filename="projects/dummy-en.zip",
            token=None,
            revision=None,
        ),
        mock.call(
            repo_id="mock-repo",
            filename="dummy-en.cfg",
            token=None,
            revision=None,
        ),
        mock.call(
            repo_id="mock-repo",
            filename="vocabs/dummy.zip",
            token=None,
            revision=None,
        ),
    ]
    dirpath_fi = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath_fi = os.path.join(str(dirpath_fi), "file.txt")
    assert os.path.exists(fpath_fi)
    dirpath_en = os.path.join(str(testdatadir), "projects", "dummy-en")
    fpath_en = os.path.join(str(dirpath_en), "file.txt")
    assert os.path.exists(fpath_en)
    assert copy_project_config.call_args_list == [
        mock.call("tests/huggingface-cache/dummy-fi.cfg", False),
        mock.call("tests/huggingface-cache/dummy-en.cfg", False),
    ]


@mock.patch(
    "huggingface_hub.list_repo_files",
    side_effect=HFValidationError,
)
@mock.patch(
    "huggingface_hub.hf_hub_download",
)
def test_download_list_repo_files_failed(
    hf_hub_download,
    list_repo_files,
):
    failed_result = runner.invoke(
        annif.cli.cli,
        [
            "download",
            "dummy-fi",
            "mock-repo",
        ],
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Error: Operation failed:" in failed_result.output
    assert list_repo_files.called
    assert not hf_hub_download.called


@mock.patch(
    "huggingface_hub.list_repo_files",
    return_value=[  # Mock filenames in repo
        "projects/dummy-fi.zip",
        "vocabs/dummy.zip",
        "dummy-fi.cfg",
    ],
)
@mock.patch(
    "huggingface_hub.hf_hub_download",
    side_effect=HFValidationError,
)
def test_download_hf_hub_download_failed(
    hf_hub_download,
    list_repo_files,
):
    failed_result = runner.invoke(
        annif.cli.cli,
        [
            "download",
            "dummy-fi",
            "mock-repo",
        ],
    )
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Error: Operation failed:" in failed_result.output
    assert list_repo_files.called
    assert hf_hub_download.called


def test_unzip_archive_initial(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    annif.cli_util.unzip_archive(
        os.path.join("tests", "huggingface-cache", "projects", "dummy-fi.zip"),
        force=False,
    )
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == 0  # Zero content from zip
    ts = os.path.getmtime(fpath)
    assert datetime.fromtimestamp(ts).astimezone(tz=timezone.utc) == datetime(
        1980, 1, 1, 0, 0
    ).astimezone(tz=timezone.utc)


def test_unzip_archive_no_overwrite(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    os.makedirs(dirpath, exist_ok=True)
    with open(fpath, "wt") as pf:
        print("Existing content", file=pf)

    annif.cli_util.unzip_archive(
        os.path.join("tests", "huggingface-cache", "projects", "dummy-fi.zip"),
        force=False,
    )
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == 17  # Existing content
    assert datetime.now().timestamp() - os.path.getmtime(fpath) < 1


def test_unzip_archive_overwrite(testdatadir):
    dirpath = os.path.join(str(testdatadir), "projects", "dummy-fi")
    fpath = os.path.join(str(dirpath), "file.txt")
    os.makedirs(dirpath, exist_ok=True)
    with open(fpath, "wt") as pf:
        print("Existing content", file=pf)

    annif.cli_util.unzip_archive(
        os.path.join("tests", "huggingface-cache", "projects", "dummy-fi.zip"),
        force=True,
    )
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == 0  # Zero content from zip
    ts = os.path.getmtime(fpath)
    assert datetime.fromtimestamp(ts).astimezone(tz=timezone.utc) == datetime(
        1980, 1, 1, 0, 0
    ).astimezone(tz=timezone.utc)


@mock.patch("os.path.exists", return_value=True)
@mock.patch("annif.cli_util._compute_crc32", return_value=0)
@mock.patch("shutil.copy")
def test_copy_project_config_no_overwrite(copy, _compute_crc32, exists):
    annif.cli_util.copy_project_config(
        os.path.join("tests", "huggingface-cache", "dummy-fi.cfg"), force=False
    )
    assert not copy.called


@mock.patch("os.path.exists", return_value=True)
@mock.patch("shutil.copy")
def test_copy_project_config_overwrite(copy, exists):
    annif.cli_util.copy_project_config(
        os.path.join("tests", "huggingface-cache", "dummy-fi.cfg"), force=True
    )
    assert copy.called
    assert copy.call_args == mock.call(
        "tests/huggingface-cache/dummy-fi.cfg", "projects.d/dummy-fi.cfg"
    )


def test_completion_script_generation():
    result = runner.invoke(annif.cli.cli, ["completion", "--bash"])
    assert not result.exception
    assert result.exit_code == 0
    assert "# Generated by Annif " in result.output


def test_completion_script_generation_shell_not_given():
    failed_result = runner.invoke(annif.cli.cli, ["completion"])
    assert failed_result.exception
    assert failed_result.exit_code != 0
    assert "Shell not given" in failed_result.output


def get_completions(cli, args, incomplete):
    completer = ShellComplete(cli, {}, cli.name, "_ANNIF_COMPLETE")
    completions = completer.get_completions(args, incomplete)
    return [c.value for c in completions]


def test_completion_list_commands():
    completions = get_completions(annif.cli.cli, [""], "list")
    assert completions == ["list-projects", "list-vocabs"]


def test_completion_version_option():
    completions = get_completions(annif.cli.cli, [""], "--ver")
    assert completions == ["--version"]


@mock.patch.dict(os.environ, {"ANNIF_CONFIG": "annif.default_config.TestingConfig"})
def test_completion_show_project_project_ids_all():
    completions = get_completions(annif.cli.cli, ["show-project"], "")
    assert completions == [
        "dummy-fi",
        "dummy-en",
        "dummy-private",
        "dummy-vocablang",
        "dummy-transform",
        "limit-transform",
        "ensemble",
        "noanalyzer",
        "novocab",
        "nobackend",
        "noname",
        "noparams-tfidf-fi",
        "noparams-fasttext-fi",
        "pav",
        "tfidf-fi",
        "tfidf-en",
        "fasttext-en",
        "fasttext-fi",
    ]


@mock.patch.dict(os.environ, {"ANNIF_CONFIG": "annif.default_config.TestingConfig"})
def test_completion_show_project_project_ids_dummy():
    completions = get_completions(annif.cli.cli, ["show-project"], "dummy")
    assert completions == [
        "dummy-fi",
        "dummy-en",
        "dummy-private",
        "dummy-vocablang",
        "dummy-transform",
    ]


@mock.patch.dict(os.environ, {"ANNIF_CONFIG": "annif.default_config.TestingConfig"})
def test_completion_load_vocab_vocab_ids_all():
    completions = get_completions(annif.cli.cli, ["load-vocab"], "")
    assert completions == ["dummy", "dummy-noname", "yso"]
