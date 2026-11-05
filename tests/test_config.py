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
    # the raw configuration still contains the unexpanded placeholder
    assert cfg["testproj"].raw_values()["key"] == "${ANNIF_TEST_KEY}"


def test_raw_values_toml(tmpdir, monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    conffile = tmpdir.join("projects.toml")
    conffile.write('[testproj]\nkey = "${ANNIF_TEST_KEY}"\nnumber = 42\n')
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["key"] == "secret123"
    # the raw configuration still contains the unexpanded placeholder
    raw = cfg["testproj"].raw_values()
    assert raw == {"key": "${ANNIF_TEST_KEY}", "number": 42}


def test_interpolate_env_vars_cfg_default_section(tmpdir, monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    conffile = tmpdir.join("projects.cfg")
    conffile.write("[DEFAULT]\ncommon=${ANNIF_TEST_KEY}\n" "[testproj]\nkey=value\n")
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["common"] == "secret123"


def test_interpolate_env_vars_cfg_default_interpolation_template(tmpdir, monkeypatch):
    # interpolation templates in the DEFAULT section are preserved and
    # resolved in the context of each section
    monkeypatch.setenv("ANNIF_TEST_KEY", "abc")
    conffile = tmpdir.join("projects.cfg")
    conffile.write(
        "[DEFAULT]\nname=%(language)s ${ANNIF_TEST_KEY}\n" "[testproj]\nlanguage=en\n"
    )
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["name"] == "en abc"


def test_interpolate_env_vars_cfg_default_env_value_with_percent(tmpdir, monkeypatch):
    # a percent sign in the environment variable value in the DEFAULT
    # section is preserved as-is
    monkeypatch.setenv("ANNIF_TEST_KEY", "a%b")
    conffile = tmpdir.join("projects.cfg")
    conffile.write("[DEFAULT]\nkey=${ANNIF_TEST_KEY}\n[testproj]\nlanguage=en\n")
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["key"] == "a%b"


def test_interpolate_env_vars_cfg_escaped_percent_sign(tmpdir):
    # an escaped percent sign in a CFG value is preserved as-is
    conffile = tmpdir.join("projects.cfg")
    conffile.write("[testproj]\nname=100%% ready\n")
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["name"] == "100% ready"


def test_interpolate_env_vars_cfg_env_value_with_percent(tmpdir, monkeypatch):
    # a percent sign in the environment variable value is preserved as-is
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret%123")
    conffile = tmpdir.join("projects.cfg")
    conffile.write("[testproj]\nkey=${ANNIF_TEST_KEY}\n")
    cfg = annif.config.parse_config(str(conffile))
    assert cfg["testproj"]["key"] == "secret%123"


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
    # reading the configuration does not raise; the error is raised when
    # the value is accessed
    cfg = annif.config.parse_config(str(conffile))
    with pytest.raises(ConfigurationException) as excinfo:
        _ = cfg["testproj"]["key"]
    message = excinfo.value.format_message()
    assert "ANNIF_TEST_MISSING" in message
    assert "'testproj'" in message
    assert "'key'" in message


def test_interpolate_env_vars_missing_var_toml(tmpdir, monkeypatch):
    monkeypatch.delenv("ANNIF_TEST_MISSING", raising=False)
    conffile = tmpdir.join("projects.toml")
    conffile.write('[testproj]\nkey = "${ANNIF_TEST_MISSING}"\n')
    # reading the configuration does not raise; the error is raised when
    # the value is accessed
    cfg = annif.config.parse_config(str(conffile))
    with pytest.raises(ConfigurationException) as excinfo:
        _ = cfg["testproj"]["key"]
    message = excinfo.value.format_message()
    assert "ANNIF_TEST_MISSING" in message
    assert "'testproj'" in message
    assert "'key'" in message
