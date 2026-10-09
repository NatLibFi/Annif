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
    # the endpoint parameter is the full URL of the /v1/systemone endpoint
    assert mock_request.call_args.args[0] == "http://localhost:8080/v1/systemone"


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


def test_dm_rerank_state_transform(app_project):
    """The state-transform parameter is applied to the document text
    before it is sent to the reranking service."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"state-transform": "limit(10)"})
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    # the state is truncated to 10 characters
    assert payload["state"] == "test docum"


def _sigmoid(value, threshold):
    """The same sigmoid gate as in the backend, for test expectations."""
    import math

    return 1.0 / (1.0 + math.exp(-(value - threshold) / 0.1))


def test_dm_rerank_blend_scores(app_project):
    """The blended score is source * noul^model-strength * gate, where the
    gate is a sigmoid on the raw noul score and the source score is used
    as returned (no per-document normalization)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.1},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    by_id = {int(s.subject_id): s.score for s in suggestions}
    # default model-strength=0.5, gate-threshold=-1.0
    assert by_id[0] == pytest.approx(0.9 * 0.9**0.5 * _sigmoid(0.9, -1.0))
    assert by_id[1] == pytest.approx(0.8 * 0.1**0.5 * _sigmoid(0.1, -1.0))
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_dm_rerank_blend_reorders(app_project):
    """The noul score can reorder candidates: a low source score with a
    high noul score moves above a high source score with a low noul
    score."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    suggestions = [
        SubjectSuggestion(0, 0.9),
        SubjectSuggestion(1, 0.8),
        SubjectSuggestion(2, 0.7),
    ]
    noul_scores = {0: 0.01, 1: 0.99, 2: 0.5}
    blended = dm_rerank_type._blend(suggestions, noul_scores, 1.0, -0.25)
    by_id = {s.subject_id: s.score for s in blended}
    assert by_id[0] == pytest.approx(0.9 * 0.01 * _sigmoid(0.01, -0.25))
    assert by_id[1] == pytest.approx(0.8 * 0.99 * _sigmoid(0.99, -0.25))
    assert by_id[2] == pytest.approx(0.7 * 0.5 * _sigmoid(0.5, -0.25))
    # subject 1 (2nd by source) moves to the top
    assert [s.subject_id for s in blended] == [1, 2, 0]


def test_dm_rerank_blend_gate(app_project):
    """With model-strength=0 and a positive gate threshold the noul score
    acts as a precision gate: a low-noul candidate is demoted below a
    high-noul one despite the better source score."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.1},
                "1": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(
            app_project, **{"model-strength": 0.0, "gate-threshold": 0.4}
        )
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    by_id = {int(s.subject_id): s.score for s in suggestions}
    # subject 0: the sigmoid gate is nearly closed
    # (sigmoid((0.1 - 0.4) / 0.1) ~= 0.047), so it is demoted
    assert by_id[0] == pytest.approx(0.9 * _sigmoid(0.1, 0.4))
    assert by_id[1] == pytest.approx(0.8 * _sigmoid(0.9, 0.4))
    assert [int(s.subject_id) for s in suggestions] == [1, 0]
    assert by_id[0] < 0.05


def test_dm_rerank_blend_missing_noul(app_project):
    """A candidate without a noul score keeps its plain source score, so a
    missing decision model degrades to the source ranking."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    by_id = {int(s.subject_id): s.score for s in suggestions}
    # subject 1 has no noul score -> plain source score
    assert by_id[1] == pytest.approx(0.8)
    assert by_id[0] == pytest.approx(0.9 * 0.9**0.5 * _sigmoid(0.9, -1.0))
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_dm_rerank_max_candidates(app_project):
    """With max-candidates only the top N candidates are sent to the
    decision model; the rest are excluded from the output."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"max-candidates": 1})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    # only the top candidate is asked
    assert set(payload["questions"]) == {"0"}
    # the other candidate is dropped from the output
    assert sorted(int(s.subject_id) for s in result[0]) == [0]


def test_dm_rerank_max_candidates_zero_scores_all(app_project):
    """The default max-candidates=0 sends all candidates to the model."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.8},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert set(payload["questions"]) == {"0", "1"}


def test_dm_rerank_blend_pure_source(app_project):
    """With model-strength=0 and the default negative gate threshold the
    blend approximates the pure source order (the floor of the search
    space)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.05},
                "1": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"model-strength": 0.0})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    by_id = {int(s.subject_id): s.score for s in suggestions}
    # both gates are near 1.0 (default threshold -1.0 is far below both
    # noul scores), so the scores are essentially the source scores
    assert by_id[0] == pytest.approx(0.9, abs=0.05)
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


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


def test_dm_rerank_instruction_invalid_template(app_project):
    """An instruction with an unknown placeholder or malformed braces is
    rejected with a ConfigurationException instead of leaking a KeyError
    or ValueError from str.format during suggestion."""
    for bad in (
        "Is {label} about {unknown}?",
        "Is {label} a {subject}?",
        "Is {label} about {0}?",
    ):
        with (
            unittest.mock.patch("requests.post") as mock_request,
            _mock_source(app_project, [(0, 0.9)]),
        ):
            dm_rerank = _make_backend(app_project, instruction=bad)
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


def test_dm_rerank_hyperopt(app_project):
    """The hyperparameter optimizer queries the decision model once per
    document and searches the model-strength / gate-threshold blend
    parameters over the cached scores."""
    from annif.corpus import DocumentList

    corpus = DocumentList(
        [
            Document(text="a test document about dummies"),
            Document(text="another test document"),
        ]
    )

    def mock_suggest(documents, params=None):
        # the mock source returns the same suggestions for every document
        rows = [
            [SubjectSuggestion(0, 0.9), SubjectSuggestion(1, 0.8)] for _ in documents
        ]
        return SuggestionBatch.from_sequence(rows, app_project.subjects, limit=100)

    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch.object(
            app_project.registry,
            "get_project",
            return_value=unittest.mock.Mock(
                is_trained=True,
                suggest=mock_suggest,
                initialize=unittest.mock.Mock(),
            ),
        ),
    ):
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.5},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        optimizer = dm_rerank.get_hp_optimizer(corpus, metric="NDCG")
        recommendation = optimizer.optimize(n_trials=3, n_jobs=1, results_file=None)

    # the decision model is queried once per document in the corpus
    assert mock_request.call_count == 2
    # the recommendation contains both blend parameters and a score
    assert "model-strength=" in recommendation.lines[0]
    assert "gate-threshold=" in recommendation.lines[1]
    assert recommendation.score is not None
