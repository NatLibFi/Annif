"""Unit tests for Annif utility functions"""

import importlib.metadata
import logging

import pytest

import annif.util
from annif.exception import ConfigurationException

from .util import umask_context


class MockSaveable:
    def save(self, filename):
        with open(filename, "w") as f:
            f.write("data")


# parametrize test to verify that file permissions are set correctly
# using commonly used umask values
@pytest.mark.parametrize(
    "umask,mode", [(0o002, 0o664), (0o022, 0o644), (0o027, 0o640), (0o077, 0o600)]
)
def test_atomic_save(tmpdir, umask, mode):
    obj = MockSaveable()
    dirname = str(tmpdir)
    filename = "myfile"

    with umask_context(umask):
        annif.util.atomic_save(obj, dirname, filename)

    final_path = tmpdir.join(filename)
    assert final_path.exists()

    # verify file content
    assert final_path.read_text(encoding="utf-8") == "data"

    # verify file permissions
    assert final_path.stat().mode & 0o777 == mode


def test_boolean():
    inputs = ["1", "0", "true", "false", "TRUE", "FALSE", "Yes", "No", True, False]
    outputs = [True, False, True, False, True, False, True, False, True, False]

    for input, output in zip(inputs, outputs):
        assert annif.util.boolean(input) == output


def test_metric_code():
    inputs = ["F1 score (doc avg)", "NDCG@10", "True positives"]
    outputs = ["F1_score_doc_avg", "NDCG@10", "True_positives"]

    for input, output in zip(inputs, outputs):
        assert annif.util.metric_code(input) == output


def test_user_agent_header():
    version = importlib.metadata.version("annif")
    headers = annif.util.user_agent_header()
    assert headers == {"User-Agent": f"Annif/{version}"}
    # each call returns a fresh dict, so callers can add their own headers
    headers["Authorization"] = "Bearer test"
    assert annif.util.user_agent_header() == {"User-Agent": f"Annif/{version}"}


def test_interpolate_env_vars_single(monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "secret123")
    assert annif.util.interpolate_env_vars("${ANNIF_TEST_KEY}", "proj", "opt") == (
        "secret123"
    )


def test_interpolate_env_vars_multiple(monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "abc")
    monkeypatch.setenv("ANNIF_TEST_KEY2", "def")
    assert (
        annif.util.interpolate_env_vars(
            "${ANNIF_TEST_KEY}-${ANNIF_TEST_KEY2}", "proj", "opt"
        )
        == "abc-def"
    )


def test_interpolate_env_vars_embedded(monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "abc")
    assert (
        annif.util.interpolate_env_vars("http://x/${ANNIF_TEST_KEY}/y", "proj", "opt")
        == "http://x/abc/y"
    )


def test_interpolate_env_vars_empty_value(monkeypatch):
    monkeypatch.setenv("ANNIF_TEST_KEY", "")
    assert annif.util.interpolate_env_vars("${ANNIF_TEST_KEY}", "proj", "opt") == ""


def test_interpolate_env_vars_missing_raises(monkeypatch):
    monkeypatch.delenv("ANNIF_TEST_MISSING", raising=False)
    with pytest.raises(ConfigurationException) as excinfo:
        annif.util.interpolate_env_vars("${ANNIF_TEST_MISSING}", "proj", "opt")
    assert "ANNIF_TEST_MISSING" in str(excinfo.value)
    assert "proj" in excinfo.value.format_message()
    assert "'opt'" in excinfo.value.format_message()


def test_interpolate_env_vars_no_placeholder():
    # a plain dollar sign and invalid variable names are left unchanged
    assert annif.util.interpolate_env_vars("no change $here", "proj", "opt") == (
        "no change $here"
    )
    assert (
        annif.util.interpolate_env_vars("${not an env var}", "proj", "opt")
        == "${not an env var}"
    )
    assert annif.util.interpolate_env_vars("$100", "proj", "opt") == "$100"


def test_parse_sources():
    assert annif.util.parse_sources("a:1,b:3") == [("a", 0.25), ("b", 0.75)]
    # the weight defaults to 1.0
    assert annif.util.parse_sources("a, b") == [("a", 0.5), ("b", 0.5)]


def test_parse_sources_zero_weight(restore_duplicate_filters, caplog):
    # a single source with weight 0 is equivalent to the same source without one
    with caplog.at_level(logging.WARNING):
        assert annif.util.parse_sources("tfidf:0") == [("tfidf", 1.0)]
    assert "weight 0 for source 'tfidf' in sources 'tfidf:0'" in caplog.text
    assert f"using {annif.util.MIN_SOURCE_WEIGHT} instead" in caplog.text


def test_parse_sources_all_zero_weights(restore_duplicate_filters):
    assert annif.util.parse_sources("a:0,b:0") == [("a", 0.5), ("b", 0.5)]


@pytest.mark.parametrize(
    "sourcedef, expected",
    [
        ("a:2,b:-1", [("a", 2.0), ("b", annif.util.MIN_SOURCE_WEIGHT)]),
        ("a:1,b:-1", [("a", 1.0), ("b", annif.util.MIN_SOURCE_WEIGHT)]),
        ("a:0,b:3", [("a", annif.util.MIN_SOURCE_WEIGHT), ("b", 3.0)]),
        ("a:1,b:nan", [("a", 1.0), ("b", annif.util.MIN_SOURCE_WEIGHT)]),
        ("a:1,b:inf", [("a", 1.0), ("b", annif.util.MIN_SOURCE_WEIGHT)]),
        ("a:-inf,b:3", [("a", annif.util.MIN_SOURCE_WEIGHT), ("b", 3.0)]),
    ],
)
def test_parse_sources_replaces_invalid_weight(
    sourcedef, expected, restore_duplicate_filters, caplog
):
    # a weight that is <= 0 or not finite is replaced by the minimum weight, and
    # the weights are normalized as usual
    total = sum(weight for _, weight in expected)
    with caplog.at_level(logging.WARNING):
        result = annif.util.parse_sources(sourcedef)
    assert result == [(src, pytest.approx(weight / total)) for src, weight in expected]
    assert sum(weight for _, weight in result) == pytest.approx(1.0)
    assert "is not a positive finite number" in caplog.text


@pytest.mark.parametrize("sourcedef", ["a:nan", "a:inf", "a:-inf", "a:1e400"])
def test_parse_sources_non_finite_weight(sourcedef, restore_duplicate_filters, caplog):
    # a single source with a weight that is not finite gets the minimum weight,
    # which normalizes to 1.0, as for a weight of 0
    with caplog.at_level(logging.WARNING):
        assert annif.util.parse_sources(sourcedef) == [("a", 1.0)]
    weight = sourcedef.split(":")[1]
    assert f"weight {weight} for source 'a' in sources '{sourcedef}'" in caplog.text
    assert f"using {annif.util.MIN_SOURCE_WEIGHT} instead" in caplog.text
    assert len(caplog.records) == 1


@pytest.mark.parametrize("sourcedef", ["tfidf:x", "a:1,b:x"])
def test_parse_sources_not_a_number_raises(sourcedef):
    with pytest.raises(ConfigurationException) as excinfo:
        annif.util.parse_sources(sourcedef)
    assert "is not a number" in str(excinfo.value)
    assert sourcedef in str(excinfo.value)
