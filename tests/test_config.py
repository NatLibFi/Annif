"""Unit tests for configuration handling in Annif"""

import logging

import pytest

import annif.config
from annif.exception import ConfigurationException


def test_check_config_exists_cfg():
    cfg = annif.config.check_config("tests/projects.cfg")
    assert cfg == "tests/projects.cfg"


def test_check_config_exists_toml():
    cfg = annif.config.check_config("tests/projects.toml")
    assert cfg == "tests/projects.toml"


def test_check_config_exists_directory():
    cfg = annif.config.check_config("tests/projects.d")
    assert cfg == "tests/projects.d"


def test_check_config_not_exists(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = annif.config.check_config("tests/notfound.cfg")
    assert cfg is None
    assert (
        'Project configuration file or directory "tests/notfound.cfg" ' + "is missing"
        in caplog.text
    )


def test_find_config_exists_default(monkeypatch):
    # temporarily chdir to the tests directory
    monkeypatch.chdir("tests")
    cfg = annif.config.find_config()
    assert cfg == "projects.cfg"


def test_find_config_not_exists_default(monkeypatch, caplog):
    # temporarily chdir to the tests/corpora directory
    monkeypatch.chdir("tests/corpora")
    with caplog.at_level(logging.WARNING):
        cfg = annif.config.find_config()
    assert cfg is None
    assert "Could not find project configuration " in caplog.text


def test_parse_config_cfg_nondefault():
    cfg = annif.config.parse_config("tests/projects.cfg")
    assert isinstance(cfg, annif.config.AnnifConfigCFG)
    assert len(cfg.project_ids) == 17
    assert cfg["dummy-fi"] is not None


def test_parse_config_cfg_default(monkeypatch):
    # temporarily chdir to the tests directory
    monkeypatch.chdir("tests")
    cfg = annif.config.parse_config("")
    assert isinstance(cfg, annif.config.AnnifConfigCFG)
    assert len(cfg.project_ids) == 17
    assert cfg["dummy-fi"] is not None


def test_parse_config_toml():
    cfg = annif.config.parse_config("tests/projects.toml")
    assert isinstance(cfg, annif.config.AnnifConfigTOML)
    assert len(cfg.project_ids) == 2
    assert cfg["dummy-fi-toml"] is not None


def test_parse_config_toml_failed(tmpdir):
    conffile = tmpdir.join("projects.toml")
    conffile.write("[section]\nkey=unquoted\n")
    with pytest.raises(ConfigurationException) as excinfo:
        annif.config.parse_config(str(conffile))
    assert "Invalid value" in str(excinfo.value)


def test_parse_config_directory():
    cfg = annif.config.parse_config("tests/projects.d")
    assert isinstance(cfg, annif.config.AnnifConfigDirectory)
    assert len(cfg.project_ids) == 2 + 17  # 0-projects.toml + 1-projects.cfg
    assert cfg["dummy-fi"] is not None
    assert cfg["dummy-fi-toml"] is not None
    assert list(cfg.project_ids)[0] == "dummy-fi-toml"  # First in 0-projects.toml
    assert list(cfg.project_ids)[2] == "dummy-fi"  # First in 1-projects.cfg


def test_parse_config_directory_duplicated_project(tmpdir):
    conffile = tmpdir.join("projects-1.toml")
    conffile.write("[duplicated]\nkey='value'\n")
    conffile = tmpdir.join("projects-2.cfg")
    conffile.write("[duplicated]\nkey=value\n")
    with pytest.raises(ConfigurationException) as excinfo:
        annif.config.parse_config(str(tmpdir))
    assert 'project ID "duplicated" already exists' in str(excinfo.value)


def test_interpolate_env_vars_cfg(tmpdir, monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    conffile = tmpdir.join("projects.cfg")
    conffile.write(
        "[testproj]\nkey=${ANNIF_TEST_KEY}\nurl=http://x/${ANNIF_TEST_KEY}/y\n"
    )
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["key"] == "secret123"
    assert cfg["testproj"]["url"] == "http://x/secret123/y"


def test_interpolate_env_vars_cfg_default_section(tmpdir, monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    conffile = tmpdir.join("projects.cfg")
    conffile.write("[DEFAULT]\ncommon=${ANNIF_TEST_KEY}\n" "[testproj]\nkey=value\n")
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["common"] == "secret123"


def test_interpolate_env_vars_toml(tmpdir, monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    conffile = tmpdir.join("projects.toml")
    conffile.write(
        "[testproj]\n"
        'key = "${ANNIF_TEST_KEY}"\n'
        'embedded = "http://x/${ANNIF_TEST_KEY}/y"\n'
        "number = 42\n"
        "flag = false\n"
    )
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["key"] == "secret123"
    assert cfg["testproj"]["embedded"] == "http://x/secret123/y"
    assert cfg["testproj"]["number"] == 42
    assert cfg["testproj"]["flag"] is False


def test_interpolate_env_vars_directory(tmpdir, monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    conffile = tmpdir.join("projects-1.toml")
    conffile.write('[proj-toml]\nkey = "${ANNIF_TEST_KEY}"\n')
    conffile = tmpdir.join("projects-2.cfg")
    conffile.write("[proj-cfg]\nkey=${ANNIF_TEST_KEY}\n")
    cfg = annif.config.parse_config(str(tmpdir))
    assert cfg["proj-toml"]["key"] == "secret123"
    assert cfg["proj-cfg"]["key"] == "secret123"


def test_interpolate_env_vars_missing_var_cfg(tmpdir, monkeypatch):
    monkeypatch.delenv("ANNIF_TEST_MISSING", raising=False)
    conffile = tmpdir.join("projects.cfg")
    conffile.write("[testproj]\nkey=${ANNIF_TEST_MISSING}\n")
    with pytest.raises(ConfigurationException) as excinfo:
        annif.config.parse_config(str(conffile))
    message = excinfo.value.format_message()
    assert "ANNIF_TEST_MISSING" in message
    assert "'testproj'" in message
    assert "'key'" in message


def test_interpolate_env_vars_missing_var_toml(tmpdir, monkeypatch):
    monkeypatch.delenv("ANNIF_TEST_MISSING", raising=False)
    conffile = tmpdir.join("projects.toml")
    conffile.write('[testproj]\nkey = "${ANNIF_TEST_MISSING}"\n')
    with pytest.raises(ConfigurationException) as excinfo:
        annif.config.parse_config(str(conffile))
    message = excinfo.value.format_message()
    assert "ANNIF_TEST_MISSING" in message
    assert "'testproj'" in message
    assert "'key'" in message
