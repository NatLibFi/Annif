"""Unit tests for projects in Annif"""

import logging
from datetime import datetime, timedelta, timezone

import pytest

import annif.backend.dummy
import annif.project
from annif.exception import ConfigurationException, NotSupportedException
from annif.project import Access


def test_create_project_wrong_access(registry):
    with pytest.raises(ConfigurationException):
        annif.project.AnnifProject(
            "example",
            {"name": "Example", "language": "en", "access": "invalid"},
            ".",
            registry,
        )


def test_get_project_en(registry):
    project = registry.get_project("dummy-en")
    assert project.project_id == "dummy-en"
    assert project.language == "en"
    assert project.analyzer.name == "snowball"
    assert project.analyzer.param == "english"
    assert project.access == Access.hidden
    assert isinstance(project.backend, annif.backend.dummy.DummyBackend)


def test_get_project_fi(registry):
    project = registry.get_project("dummy-fi")
    assert project.project_id == "dummy-fi"
    assert project.language == "fi"
    assert project.analyzer.name == "snowball"
    assert project.analyzer.param == "finnish"
    assert project.vocab_lang == "fi"
    assert project.access == Access.public
    assert isinstance(project.backend, annif.backend.dummy.DummyBackend)


def test_get_project_dummy_private(registry):
    project = registry.get_project("dummy-private")
    assert project.project_id == "dummy-private"
    assert project.language == "en"
    assert project.analyzer.name == "snowball"
    assert project.analyzer.param == "english"
    assert project.access == Access.private
    assert isinstance(project.backend, annif.backend.dummy.DummyBackend)


def test_get_project_dummy_vocablang(registry):
    project = registry.get_project("dummy-vocablang")
    assert project.project_id == "dummy-vocablang"
    assert project.language == "en"
    assert project.analyzer.name == "snowball"
    assert project.analyzer.param == "english"
    # project uses the dummy vocab, with language overridden to Finnish
    assert project.vocab.vocab_id == "dummy"
    assert project.vocab_lang == "fi"
    assert project.access == Access.public
    assert isinstance(project.backend, annif.backend.dummy.DummyBackend)


def test_get_project_fi_dump(registry):
    project = registry.get_project("dummy-fi")
    pdump = project.dump()
    assert pdump == {
        "project_id": "dummy-fi",
        "name": "Dummy Finnish",
        "language": "fi",
        "backend": {
            "backend_id": "dummy",
        },
        "is_trained": True,
        "modification_time": None,
    }


def test_get_project_nonexistent(registry):
    with pytest.raises(ValueError):
        registry.get_project("nonexistent")


def test_get_project_noanalyzer(registry):
    project = registry.get_project("noanalyzer")
    with pytest.raises(ConfigurationException):
        project.analyzer


def test_get_project_novocab(registry):
    project = registry.get_project("novocab")
    with pytest.raises(ConfigurationException):
        project.vocab


def test_get_project_nobackend(registry):
    project = registry.get_project("nobackend")
    with pytest.raises(ConfigurationException):
        project.backend


def test_get_project_noname(registry):
    project = registry.get_project("noname")
    assert project.name == project.project_id


def test_get_project_default_params_tfidf(registry):
    project = registry.get_project("noparams-tfidf-fi")
    expected_default_params = {"limit": 100}  # From AnnifBackend class
    actual_params = project.backend.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_get_project_default_params_fasttext(registry):
    pytest.importorskip("annif.backend.fasttext")
    project = registry.get_project("noparams-fasttext-fi")
    expected_default_params = {
        "limit": 100,  # From AnnifBackend class
        "dim": 100,  # Rest from FastTextBackend class
        "lr": 0.25,
        "epoch": 5,
        "loss": "hs",
    }
    actual_params = project.backend.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_get_project_invalid_config_file():
    app = annif.create_app(
        config_name="annif.default_config.TestingInvalidProjectsConfig"
    )
    with app.app_context():
        with pytest.raises(ConfigurationException):
            annif.registry.get_project("duplicatedvocab")


def test_project_load_vocabulary_tfidf(registry, subject_file, testdatadir):
    project = registry.get_project("tfidf-fi")
    project.vocab.load_vocabulary(subject_file)
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0


def test_project_tfidf_is_not_trained(registry):
    project = registry.get_project("tfidf-fi")
    assert not project.is_trained


def test_project_train_tfidf(registry, document_corpus, testdatadir):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    assert testdatadir.join("projects/tfidf-fi/tfidf-index").exists()
    assert testdatadir.join("projects/tfidf-fi/tfidf-index").size() > 0


def test_project_tfidf_is_trained(registry):
    project = registry.get_project("tfidf-fi")
    assert project.is_trained


def test_project_tfidf_modification_time(registry):
    project = registry.get_project("tfidf-fi")
    assert datetime.now(timezone.utc) - project.modification_time < timedelta(1)


def test_project_train_tfidf_nodocuments(registry, empty_corpus):
    project = registry.get_project("tfidf-fi")
    with pytest.raises(NotSupportedException) as excinfo:
        project.train(empty_corpus)
    assert "Cannot train tfidf project with no documents" in str(excinfo.value)


def test_project_learn(registry, tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.tsv").write("<http://example.org/none>\tnone")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.tsv").write("<http://example.org/dummy>\tdummy")

    project = registry.get_project("dummy-fi")
    docdir = annif.corpus.DocumentDirectory(str(tmpdir), project.subjects, "en")
    project.learn(docdir)
    result = project.suggest("this is some text")
    assert len(result) == 1
    hits = result.as_list()
    assert hits[0].subject_id == project.subjects.by_uri("http://example.org/none")
    assert hits[0].score == 1.0


def test_project_learn_not_supported(registry, tmpdir):
    tmpdir.join("doc1.txt").write("doc1")
    tmpdir.join("doc1.tsv").write("<http://example.org/key1>\tkey1")
    tmpdir.join("doc2.txt").write("doc2")
    tmpdir.join("doc2.tsv").write("<http://example.org/key2>\tkey2")

    project = registry.get_project("tfidf-fi")
    docdir = annif.corpus.DocumentDirectory(str(tmpdir), project.subjects, "en")
    with pytest.raises(NotSupportedException):
        project.learn(docdir)


def test_project_load_vocabulary_fasttext(registry, subject_file, testdatadir):
    pytest.importorskip("annif.backend.fasttext")
    project = registry.get_project("fasttext-fi")
    project.vocab.load_vocabulary(subject_file)
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0


def test_project_train_fasttext(registry, document_corpus, testdatadir):
    pytest.importorskip("annif.backend.fasttext")
    project = registry.get_project("fasttext-fi")
    project.train(document_corpus)
    assert testdatadir.join("projects/fasttext-fi/fasttext-model").exists()
    assert testdatadir.join("projects/fasttext-fi/fasttext-model").size() > 0


def test_project_suggest(registry):
    project = registry.get_project("dummy-en")
    result = project.suggest("this is some text")
    assert len(result) == 1
    hits = result.as_list()
    assert hits[0].subject_id == project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == 1.0


def test_project_suggest_transform_limit(registry):
    project = registry.get_project("limit-transform")
    result = project.suggest("this is some text")
    assert len(result) == 0


def test_project_suggest_batch(registry, fulltext_corpus):
    project = registry.get_project("dummy-en")
    result = project.suggest_batch(fulltext_corpus)
    assert len(result) == 28  # Number of documents
    first_doc_hits = result[0].as_list()
    assert len(first_doc_hits) == 1
    assert first_doc_hits[0].subject_id == project.subjects.by_uri(
        "http://example.org/dummy"
    )
    assert first_doc_hits[0].score == 1.0


def test_project_suggest_batch_transform_limit(registry, fulltext_corpus):
    project = registry.get_project("limit-transform")
    result = project.suggest_batch(fulltext_corpus)
    assert len(result) == 28  # Number of documents
    assert len(result[0]) == 0


def test_project_train_state_not_available(registry, caplog):
    project = registry.get_project("dummy-vocablang")
    project.backend.is_trained = None
    with caplog.at_level(logging.WARNING):
        result = project.suggest("this is some text")
    assert project.is_trained is None
    assert len(result) == 1
    hits = result.as_list()
    assert hits[0].subject_id == project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == 1.0
    assert "Could not get train state information" in caplog.text


def test_project_transform_text_pass_through(registry):
    project = registry.get_project("dummy-transform")
    assert project.transform.transform_text("this is some text") == "this is some text"


def test_project_not_initialized(registry):
    project = registry.get_project("dummy-en")
    assert not project.initialized


def test_project_initialized(app_with_initialize):
    with app_with_initialize.app_context():
        project = annif.registry.get_project("dummy-en")
    assert project.initialized
    assert project.backend.initialized


def test_project_file_not_found():
    app = annif.create_app(config_name="annif.default_config.TestingNoProjectsConfig")
    with app.app_context():
        with pytest.raises(ValueError):
            annif.registry.get_project("dummy-en")


def test_project_file_toml():
    app = annif.create_app(config_name="annif.default_config.TestingTOMLConfig")
    with app.app_context():
        assert len(annif.registry.get_projects()) == 2
        assert annif.registry.get_project("dummy-fi-toml").project_id == "dummy-fi-toml"
        assert annif.registry.get_project("dummy-en-toml").project_id == "dummy-en-toml"


def test_project_directory():
    app = annif.create_app(config_name="annif.default_config.TestingDirectoryConfig")
    with app.app_context():
        assert len(annif.registry.get_projects()) == 18 + 2
        assert annif.registry.get_project("dummy-fi").project_id == "dummy-fi"
        assert annif.registry.get_project("dummy-fi-toml").project_id == "dummy-fi-toml"
