"""Unit tests for the dm_rerank backend in Annif"""

import unittest.mock

import pytest
import requests.exceptions

import annif.backend
from annif.corpus import Document
from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.vocab import Subject


def _make_backend(project, **params):
    base = {
        "sources": "dummy-en",
        "endpoint": "http://127.0.0.1:8700",
    }
    base.update(params)
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    return dm_rerank_type(backend_id="dm_rerank", config_params=base, project=project)


def _mock_source(app_project, subject_ids_and_scores, is_trained=True):
    """Return a context manager mocking registry.get_project to serve a source
    project whose suggest() returns a real SuggestionBatch."""
    src_batch = SuggestionBatch.from_sequence(
        [[SubjectSuggestion(sid, score) for sid, score in subject_ids_and_scores]],
        app_project.subjects,
        limit=100,
    )
    mock_proj = unittest.mock.Mock()
    mock_proj.is_trained = is_trained
    mock_proj.suggest.return_value = src_batch
    return unittest.mock.patch.object(
        app_project.registry, "get_project", return_value=mock_proj
    )


def test_dm_rerank_suggest_request_timeout(app_project):
    """The request uses the timeout parameter (default 60 seconds)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])
        assert mock_request.call_args.kwargs["timeout"] == 60.0

        dm_rerank = _make_backend(app_project, timeout=5)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])
        assert mock_request.call_args.kwargs["timeout"] == 5.0


def test_dm_rerank_suggest_request_shape(app_project):
    """The request payload sent to the reranking service has the expected
    shape."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.7},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            dm_rerank.suggest([Document(text="test document")])

    # verify the request payload shape sent to the reranking service
    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "test document"
    # the model parameter defaults to empty and is then omitted from the
    # payload
    assert "model" not in payload
    # the default instruction is the "sharp" centrality predicate
    expected = (
        "Is '{label}' a central subject of this document - one a "
        "librarian would assign as a primary heading - not merely a "
        "passing or incidental mention?"
    )
    assert payload["questions"]["0"]["instructions"] == expected.format(label="dummy")
    assert payload["questions"]["1"]["instructions"] == expected.format(label="none")
    # the endpoint parameter is used as the base URL for /v1/systemone
    assert mock_request.call_args.args[0] == "http://127.0.0.1:8700/v1/systemone"


def test_dm_rerank_suggest_model_included_when_set(app_project):
    """A non-empty model parameter is included in the request payload."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, model="some-model")
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["model"] == "some-model"


def test_dm_rerank_state_rules_prefix(app_project):
    """With state-rules enabled the built-in indexing-policy rules are
    prepended to the document state."""
    from annif.backend.dm_rerank import STATE_RULES

    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"state-rules": "true"})
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == STATE_RULES + "test document"


def test_dm_rerank_state_prefix(app_project):
    """A custom state-prefix is prepended to the document state."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"state-prefix": "Custom rule. "})
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "Custom rule. test document"


def test_dm_rerank_state_prefix_overrides_state_rules(app_project):
    """state-prefix takes precedence when state-rules is also enabled."""
    from annif.backend.dm_rerank import STATE_RULES

    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(
            app_project, **{"state-rules": "true", "state-prefix": "Custom rule. "}
        )
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "Custom rule. test document"
    assert not payload["state"].startswith(STATE_RULES)


def test_dm_rerank_blend_scores(app_project):
    """In blend mode the score is a min-max normalized linear combination
    of the source and noul scores. A candidate that is worst in both
    source and noul scores gets exactly 0.0 and is dropped by the
    SuggestionBatch, like in rerank mode."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.1},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"blend-alpha": 0.5})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    # subject 1 scores 0.5*0.0 + 0.5*0.0 = 0.0 -> dropped
    assert [int(s.subject_id) for s in suggestions] == [0]
    assert suggestions[0].score == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)


def test_dm_rerank_blend_reorders(app_project):
    """A low source score with a high noul score can move above a
    higher source score with a low noul score."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.1},
                "1": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"blend-alpha": 0.4})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    assert len(suggestions) == 2
    by_id = {int(s.subject_id): s.score for s in suggestions}
    # source norms: 0: 1.0, 1: 0.0; noul norms: 0: 0.0, 1: 1.0
    assert by_id[0] == pytest.approx(0.4 * 1.0 + 0.6 * 0.0)  # 0.4
    assert by_id[1] == pytest.approx(0.4 * 0.0 + 0.6 * 1.0)  # 0.6
    # subject 1 (lower source, higher noul) moves above subject 0
    assert [int(s.subject_id) for s in suggestions] == [1, 0]


def test_dm_rerank_blend_missing_noul(app_project):
    """A candidate without a noul score counts as 0.0 in the blend."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"blend-alpha": 0.5})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    # subject 1: 0.5*0.0 + 0.5*0.0 = 0.0 -> dropped by SuggestionBatch
    assert [int(s.subject_id) for s in suggestions] == [0]
    assert suggestions[0].score == pytest.approx(1.0)


def test_dm_rerank_blend_alpha_extremes(app_project):
    """alpha=1.0 is pure source order, alpha=0.0 pure noul order (the
    losing candidate scores 0.0 and is dropped)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.1},
                "1": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        # alpha = 1.0: source best (0) wins, subject 1 scores 0.0 -> dropped
        dm_rerank = _make_backend(app_project, **{"blend-alpha": 1.0})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])
        assert [int(s.subject_id) for s in list(result[0])] == [0]

        # alpha = 0.0: noul best (1) wins, subject 0 scores 0.0 -> dropped
        dm_rerank = _make_backend(app_project, **{"blend-alpha": 0.0})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])
        assert [int(s.subject_id) for s in list(result[0])] == [1]


def test_dm_rerank_suggest_request_error(app_project):
    """A failed reranking request raises OperationFailedException after all
    retries have been exhausted."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        dm_rerank = _make_backend(app_project, retries=1)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])

        # 1 initial attempt + 1 retry
        assert mock_request.call_count == 2


def test_dm_rerank_suggest_retries_on_failure(app_project):
    """A transient failure is retried and a later success is returned."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep") as mock_sleep,
    ):
        good_response = unittest.mock.Mock()
        good_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.side_effect = [
            requests.exceptions.ConnectionError("transient"),
            good_response,
        ]

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = dm_rerank.suggest([Document(text="test document")])

    assert mock_request.call_count == 2
    mock_sleep.assert_called_once()
    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] == [0]


def test_dm_rerank_suggest_no_retries(app_project):
    """With retries=0 a single failure raises immediately."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        dm_rerank = _make_backend(app_project, retries=0)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])

    assert mock_request.call_count == 1


def test_dm_rerank_suggest_json_error(app_project):
    """A non-JSON reranking response raises OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.side_effect = ValueError("not json")
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_custom_instruction(app_project):
    """A custom instruction template replaces the default proposition."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(
            app_project,
            instruction="Does this document relate to {label}?",
        )
        with _mock_source(app_project, [(0, 0.9)]):
            result = dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["questions"]["0"]["instructions"] == (
        "Does this document relate to dummy?"
    )
    assert [int(s.subject_id) for s in result[0]] == [0]


def test_dm_rerank_instruction_missing_placeholder(app_project):
    """An instruction without the {label} placeholder is rejected."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        _mock_source(app_project, [(0, 0.9)]),
    ):
        dm_rerank = _make_backend(app_project, instruction="No label here")
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest([Document(text="test document")])

    mock_request.assert_not_called()


def test_dm_rerank_train_not_supported(app_project):
    """Training the dm_rerank backend raises NotSupportedException."""
    dm_rerank = _make_backend(app_project)
    corpus = unittest.mock.Mock()
    with pytest.raises(NotSupportedException):
        dm_rerank.train(corpus)


def test_dm_rerank_is_trained(app_project):
    """is_trained is True only when every source is trained."""
    dm_rerank = _make_backend(app_project)
    with _mock_source(app_project, [], is_trained=True):
        assert dm_rerank.is_trained is True
    with _mock_source(app_project, [], is_trained=False):
        assert dm_rerank.is_trained is False


def test_dm_rerank_label_project_language(app_project):
    """Label selection prefers the project language, then any label, then
    notation."""
    proj = unittest.mock.Mock()
    proj.language = "fi"
    proj.datadir = "/tmp"
    proj.subjects = {
        0: Subject(uri="u0", labels={"en": "Alpha"}, notation=None),
        1: Subject(uri="u1", labels=None, notation="42.42"),
        2: Subject(uri="u2", labels={"fi": "beta-fi", "en": "Beta"}, notation=None),
        3: Subject(uri="u3", labels=None, notation=None),
    }
    dm_rerank = annif.backend.get_backend("dm_rerank")(
        backend_id="dm_rerank", config_params={"sources": "dummy-en"}, project=proj
    )

    # subject 2 has a fi label -> use it
    assert dm_rerank._label_for_subject(2) == "beta-fi"
    # subject 0 only has en -> fall back to that
    assert dm_rerank._label_for_subject(0) == "Alpha"
    # subject 1 has no labels -> fall back to notation
    assert dm_rerank._label_for_subject(1) == "42.42"
    # subject 3 has neither -> None (candidate skipped)
    assert dm_rerank._label_for_subject(3) is None
