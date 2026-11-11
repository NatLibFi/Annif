"""Unit tests for projects in Annif"""

import logging
from datetime import datetime, timedelta, timezone

import pytest

import annif
import annif.backend.dummy
import annif.default_config
import annif.project
from annif.corpus import Document
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
        "vocab": {
            "vocab_id": "dummy",
            "languages": ["en", "fi"],
            "size": 2,
            "loaded": True,
        },
        "vocab_language": "fi",
        "is_trained": True,
        "modification_time": None,
    }


class SourceWeightsConfig(annif.default_config.TestingConfig):
    PROJECTS_CONFIG_PATH = "tests/projects_source_weights.cfg"


@pytest.fixture(scope="module")
def source_weights_cxapp(cxapp):
    # the cxapp fixture makes sure the dummy vocab is in place; the projects with
    # different sources values live in a separate app, so that the shared
    # fixture projects (and their counts) are unchanged
    return annif.create_app(config_name="tests.test_project.SourceWeightsConfig")


def test_project_zero_source_weight_lists_like_default(
    source_weights_cxapp, restore_duplicate_filters, caplog
):
    # a zero source weight is replaced by a small positive one, so the project
    # behaves like one with the default weight
    with caplog.at_level(logging.WARNING):
        with source_weights_cxapp.test_client() as client:
            response = client.get("/v1/projects")
    assert response.status_code == 200
    assert "weight 0 for source 'dummy-fi'" in caplog.text
    projects = {proj["project_id"]: proj for proj in response.json()["projects"]}
    assert projects["weights-ref"]["is_trained"] is True
    for key in (
        "backend",
        "vocab",
        "vocab_language",
        "is_trained",
        "modification_time",
    ):
        assert projects["weights-zero"][key] == projects["weights-ref"][key]


def test_project_zero_source_weight_suggests_like_default(
    source_weights_cxapp, restore_duplicate_filters
):
    docs = [Document(text="example text")]
    with source_weights_cxapp.app.app_context():
        ref = annif.registry.get_project("weights-ref").suggest(docs)[0]
        zero = annif.registry.get_project("weights-zero").suggest(docs)[0]
    assert len(list(ref)) > 0
    assert list(zero) == list(ref)


@pytest.mark.parametrize(
    "project_id, message",
    [
        ("weights-invalid-x", "is not a number"),
        ("weights-invalid-nan", "is not a finite number"),
        ("weights-invalid-inf", "is not a finite number"),
    ],
)
def test_project_invalid_source_weight(
    source_weights_cxapp, restore_duplicate_filters, project_id, message
):
    # a source weight that is not a finite number makes the project misconfigured:
    # it is still listed, with an unknown train state, but suggest() raises
    with source_weights_cxapp.test_client() as client:
        response = client.get("/v1/projects")
    assert response.status_code == 200
    projects = {proj["project_id"]: proj for proj in response.json()["projects"]}
    assert projects[project_id]["is_trained"] is None
    assert projects[project_id]["modification_time"] is None

    docs = [Document(text="example text")]
    with source_weights_cxapp.app.app_context():
        project = annif.registry.get_project(project_id)
        with pytest.raises(ConfigurationException, match=message):
            project.suggest(docs)


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


def test_get_project_invalid_config_file():
    cxapp = annif.create_app(
        config_name="annif.default_config.TestingInvalidProjectsConfig"
    )
    with cxapp.app.app_context():
        with pytest.raises(ConfigurationException) as excinfo:
            annif.registry.get_project("duplicatedvocab")
        assert "option 'vocab' in section 'duplicatedvocab' already exists" in str(
            excinfo.value
        )


def test_get_project_invalid_vocab_arg():
    cxapp = annif.create_app(
        config_name="annif.default_config.TestingInvalid2ProjectsConfig"
    )
    with cxapp.app.app_context():
        project = annif.registry.get_project("invalid-vocab-arg")
        with pytest.raises(ConfigurationException) as excinfo:
            project.subjects
        assert "unknown vocab keyword argument foo" in str(excinfo.value)


def test_project_load_vocabulary_tfidf(registry, subject_file, testdatadir):
    project = registry.get_project("tfidf-fi")
    project.vocab.load_vocabulary(subject_file)
    assert testdatadir.join("vocabs/yso/subjects.csv").exists()
    assert testdatadir.join("vocabs/yso/subjects.csv").size() > 0


def test_project_tfidf_is_not_trained(registry):
    project = registry.get_project("tfidf-fi")
    assert not project.is_trained


def test_project_tfidf_is_not_trained_prepared_only(registry, testdatadir):
    testdatadir.join("projects/tfidf-fi").ensure("vectorizer")
    testdatadir.join("projects/tfidf-fi").ensure("dummy-tfidf-train.txt")
    project = registry.get_project("tfidf-fi")
    assert not project.is_trained


def test_project_tfidf_modification_time_prepared_only(registry, testdatadir):
    testdatadir.join("projects/tfidf-fi").ensure("vectorizer")
    testdatadir.join("projects/tfidf-fi").ensure("dummy-tfidf-train.txt")
    project = registry.get_project("tfidf-fi")
    assert project.modification_time is None


def test_project_train_tfidf(registry, document_corpus, testdatadir):
    project = registry.get_project("tfidf-fi")
    project.train(document_corpus)
    assert testdatadir.join("projects/tfidf-fi/tfidf-matrix.npz").exists()
    assert testdatadir.join("projects/tfidf-fi/tfidf-matrix.npz").size() > 0


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
    docdir = annif.corpus.DocumentDirectory(
        str(tmpdir), project.subjects, "en", require_subjects=True
    )
    project.learn(docdir)
    result = project.suggest([Document(text="this is some text")])[0]
    assert len(result) == 1
    hits = list(result)
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


def test_project_suggest(registry):
    project = registry.get_project("dummy-en")
    result = project.suggest([Document(text="this is some text")])[0]
    assert len(result) == 1
    hits = list(result)
    assert hits[0].subject_id == project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == 1.0


def test_project_suggest_transform_limit(registry):
    project = registry.get_project("limit-transform")
    result = project.suggest([Document(text="this is some text")])[0]
    assert len(result) == 0


def test_project_suggest_corpus(registry, fulltext_corpus):
    project = registry.get_project("dummy-en")
    result = list(project.suggest_corpus(fulltext_corpus))
    assert len(result) == 28  # Number of documents
    first_doc_hits = list(result[0])
    assert len(first_doc_hits) == 1
    assert first_doc_hits[0].subject_id == project.subjects.by_uri(
        "http://example.org/dummy"
    )
    assert first_doc_hits[0].score == 1.0


def test_project_suggest_corpus_transform_limit(registry, fulltext_corpus):
    project = registry.get_project("limit-transform")
    result = list(project.suggest_corpus(fulltext_corpus))
    assert len(result) == 28  # Number of documents
    assert len(result[0]) == 0


def test_project_train_state_not_available(registry, caplog):
    project = registry.get_project("dummy-vocablang")
    project.backend.is_trained = None
    with caplog.at_level(logging.WARNING):
        result = project.suggest([Document(text="this is some text")])[0]
    assert project.is_trained is None
    assert len(result) == 1
    hits = list(result)
    assert hits[0].subject_id == project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == 1.0
    assert "Could not get train state information" in caplog.text


def test_project_transform_text_pass_through(registry):
    project = registry.get_project("dummy-transform")
    doc = Document(text="this is some text")
    assert project.transform.transform_doc(doc).text == "this is some text"


def test_project_not_initialized(registry):
    project = registry.get_project("dummy-en")
    assert not project.initialized


def test_project_initialized(app_with_initialize):
    with app_with_initialize.app_context():
        project = annif.registry.get_project("dummy-en")
    assert project.initialized
    assert project.backend.initialized


def test_project_not_initialized_with_selection(app_with_initialize_fi_projects):
    with app_with_initialize_fi_projects.app_context():
        project = annif.registry.get_project("dummy-en")
    assert not project.initialized
    assert not project.backend.initialized


def test_project_initialized_with_selection(app_with_initialize_fi_projects):
    with app_with_initialize_fi_projects.app_context():
        project = annif.registry.get_project("dummy-fi")
    assert project.initialized
    assert project.backend.initialized


def test_project_file_not_found():
    cxapp = annif.create_app(config_name="annif.default_config.TestingNoProjectsConfig")
    with cxapp.app.app_context():
        with pytest.raises(ValueError):
            annif.registry.get_project("dummy-en")


def test_project_file_toml():
    cxapp = annif.create_app(config_name="annif.default_config.TestingTOMLConfig")
    with cxapp.app.app_context():
        assert len(annif.registry.get_projects()) == 2
        assert annif.registry.get_project("dummy-fi-toml").project_id == "dummy-fi-toml"
        assert annif.registry.get_project("dummy-en-toml").project_id == "dummy-en-toml"


def test_project_directory():
    cxapp = annif.create_app(config_name="annif.default_config.TestingDirectoryConfig")
    with cxapp.app.app_context():
        assert len(annif.registry.get_projects()) == 17 + 2
        assert annif.registry.get_project("dummy-fi").project_id == "dummy-fi"
        assert annif.registry.get_project("dummy-fi-toml").project_id == "dummy-fi-toml"


def test_registry_validates_missing_env_var_at_startup(tmp_path, monkeypatch):
    # a missing environment variable must be an error when the registry
    # is created, even if the option is never accessed
    monkeypatch.delenv("ANNIF_TEST_MISSING", raising=False)
    conffile = tmp_path.joinpath("projects.cfg")
    conffile.write_text(
        "[testproj]\n"
        "language=en\n"
        "name=Test\n"
        "backend=tfidf\n"
        "limit=${ANNIF_TEST_MISSING}\n"
    )
    app = annif.create_flask_app("annif.default_config.TestingConfig")
    app.config["PROJECTS_CONFIG_PATH"] = str(conffile)
    with app.app_context():
        with pytest.raises(ConfigurationException) as excinfo:
            annif.registry.initialize_projects(app)
    message = excinfo.value.format_message()
    assert "ANNIF_TEST_MISSING" in message
    assert "'testproj'" in message
    assert "'limit'" in message


def test_registry_validates_env_var_default_section(tmp_path, monkeypatch):
    # environment variable placeholders in the DEFAULT section are also
    # validated at startup
    monkeypatch.delenv("ANNIF_TEST_MISSING", raising=False)
    conffile = tmp_path.joinpath("projects.cfg")
    conffile.write_text(
        "[DEFAULT]\n"
        "limit=${ANNIF_TEST_MISSING}\n"
        "[testproj]\n"
        "language=en\n"
        "name=Test\n"
        "backend=tfidf\n"
    )
    app = annif.create_flask_app("annif.default_config.TestingConfig")
    app.config["PROJECTS_CONFIG_PATH"] = str(conffile)
    with app.app_context():
        with pytest.raises(ConfigurationException) as excinfo:
            annif.registry.initialize_projects(app)
    assert "ANNIF_TEST_MISSING" in excinfo.value.format_message()
