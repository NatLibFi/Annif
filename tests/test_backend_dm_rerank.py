"""Unit tests for the dm_rerank backend in Annif"""

import math
import unittest.mock
from datetime import datetime, timezone

import pytest
import requests.exceptions

import annif.backend
from annif.corpus import Document, SubjectSet
from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.vocab import Subject, SubjectIndexFile


def _make_backend(project, **params):
    base = {
        "sources": "dummy-en",
    }
    base.update(params)
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    return dm_rerank_type(backend_id="dm_rerank", config_params=base, project=project)


def _mock_source(
    app_project,
    subject_ids_and_scores,
    is_trained=True,
    modification_time=None,
):
    """Return a context manager mocking registry.get_project to serve a source
    project whose suggest() returns a real SuggestionBatch."""
    src_batch = SuggestionBatch.from_sequence(
        [[SubjectSuggestion(sid, score) for sid, score in subject_ids_and_scores]],
        app_project.subjects,
        limit=100,
    )
    mock_proj = unittest.mock.Mock()
    mock_proj.is_trained = is_trained
    mock_proj.modification_time = modification_time
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
    assert mock_request.call_args.args[0] == "http://localhost:9931/v1/systemone"


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


def test_dm_rerank_state_transform_gets_project(app_project):
    """The backend's project is passed to the state transform, so that
    project-dependent transforms such as filter_lang can use its language
    and analyzer."""
    dm_rerank = _make_backend(app_project, **{"state-transform": "pass"})
    transform = dm_rerank._get_state_transform(dm_rerank.params)
    assert transform.project is app_project


def test_dm_rerank_state_transform_per_call_override(app_project):
    """A per-call state-transform override is honored, even when the
    configured parameter is pass (the default)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest(
                [Document(text="test document")], {"state-transform": "limit(10)"}
            )

    payload = mock_request.call_args.kwargs["json"]
    # the override truncates the state to 10 characters
    assert payload["state"] == "test docum"


def test_dm_rerank_state_transform_override_does_not_poison_cache(app_project):
    """A per-call override does not affect subsequent calls, which keep
    using the configured transform."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest(
                [Document(text="test document")], {"state-transform": "limit(10)"}
            )
            # regular call afterwards: the configured transform (pass)
            # applies again
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "test document"


def _sigmoid(value, threshold):
    """The same sigmoid gate as in the backend, for test expectations."""
    return 1.0 / (1.0 + math.exp(-(value - threshold) / 0.1))


def _odds(value, eps=0.01):
    """The same odds-ratio term as in the backend, for test expectations."""
    return (value + eps) / (1.0 - value + eps)


def _squash(raw):
    """The same monotonic [0, 1) squash as in the backend, for test
    expectations."""
    if raw <= 0.0:
        return 0.0
    return raw / (1.0 + raw)


def test_dm_rerank_squash():
    """The backend squash maps raw scores into [0, 1): positive values
    are squashed monotonically, zero and any negative value (possible
    only with a malformed noul score from the service) are clamped to
    0.0."""
    from annif.backend.dm_rerank import _squash as backend_squash

    assert backend_squash(0.0) == 0.0
    # defensive clamp for pathological negative input
    assert backend_squash(-0.5) == 0.0
    assert backend_squash(-100.0) == 0.0
    assert backend_squash(1.0) == pytest.approx(0.5)
    assert backend_squash(0.25) == pytest.approx(0.2)
    # the squash stays in [0, 1) even for very large raw scores
    assert 0.0 <= backend_squash(10000.0) < 1.0


def test_dm_rerank_blend_scores(app_project):
    """The blended score is the monotonic squash of
    source * odds(noul)^model-strength * gate, where the odds term is
    (noul + eps) / (1 - noul + eps) and the gate is a sigmoid on the raw
    noul score."""
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
    # default model-strength=0.3, gate-threshold=-1.0: the gate is exactly
    # 1.0 (fully off) at that threshold, so only the odds-ratio term applies
    raw0 = 0.9 * _odds(0.9) ** 0.3
    raw1 = 0.8 * _odds(0.1) ** 0.3
    assert by_id[0] == pytest.approx(_squash(raw0))
    assert by_id[1] == pytest.approx(_squash(raw1))
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_dm_rerank_blend_empty_suggestions():
    """Blending an empty candidate list returns an empty list."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    assert dm_rerank_type._blend([], {}, 1.0, -0.25) == []


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
    assert by_id[0] == pytest.approx(_squash(0.9 * _odds(0.01) * _sigmoid(0.01, -0.25)))
    assert by_id[1] == pytest.approx(_squash(0.8 * _odds(0.99) * _sigmoid(0.99, -0.25)))
    assert by_id[2] == pytest.approx(_squash(0.7 * _odds(0.5) * _sigmoid(0.5, -0.25)))
    # subject 1 (2nd by source) moves to the top
    assert [s.subject_id for s in blended] == [1, 2, 0]


def test_dm_rerank_blend_boosts(app_project):
    """With model-strength > 1 the decision model boosts a confident 'yes'
    candidate above the rest and demotes a confident 'no' candidate."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    suggestions = [
        SubjectSuggestion(0, 0.4),
        SubjectSuggestion(1, 0.3),
    ]
    noul_scores = {0: 0.9, 1: 0.1}
    blended = dm_rerank_type._blend(suggestions, noul_scores, 1.5, -1.0)
    by_id = {s.subject_id: s.score for s in blended}
    # exactly-off gate (threshold -1.0), pure odds-ratio effect,
    # squashed into [0, 1)
    raw0 = 0.4 * _odds(0.9) ** 1.5
    raw1 = 0.3 * _odds(0.1) ** 1.5
    assert by_id[0] == pytest.approx(_squash(raw0))
    assert by_id[1] == pytest.approx(_squash(raw1))
    # the raw boost exceeds 1.0 but the squash keeps the output in [0, 1)
    assert raw0 > 1.0
    assert 0.0 <= by_id[1] < by_id[0] < 1.0
    # the boost reorders the candidates: the 'yes' above the 'no'
    assert [s.subject_id for s in blended] == [0, 1]


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
    assert by_id[0] == pytest.approx(_squash(0.9 * _sigmoid(0.1, 0.4)))
    assert by_id[1] == pytest.approx(_squash(0.8 * _sigmoid(0.9, 0.4)))
    assert [int(s.subject_id) for s in suggestions] == [1, 0]
    assert by_id[0] < 0.05


def test_dm_rerank_blend_missing_noul(app_project):
    """A candidate without a noul score gets its source score on the same
    squashed scale as the scored candidates, so a missing decision model
    degrades to the source ranking on a stable scale."""
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
    # subject 1 has no noul score: its raw source score is mapped through
    # the same squash so both candidates are on one scale; the gate is
    # exactly off at the default threshold -1.0
    assert by_id[1] == pytest.approx(_squash(0.8))
    assert by_id[0] == pytest.approx(_squash(0.9 * _odds(0.9) ** 0.3))
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_dm_rerank_blend_no_noul_scores(app_project):
    """When no candidate has a noul score (empty answers), every score is
    the source score mapped through the same squash as the scored
    candidates: a missing model answer equals neutral model evidence and
    the scores stay on a stable scale."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    by_id = {int(s.subject_id): s.score for s in suggestions}
    assert by_id[0] == pytest.approx(_squash(0.9))
    assert by_id[1] == pytest.approx(_squash(0.8))
    # the squash is monotonic, so the source order is preserved
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


def test_dm_rerank_reranking_promotes_beyond_limit(app_project):
    """When the output limit is smaller than the merged candidate pool, a
    low-ranked source candidate that the decision model endorses can still
    be promoted into the output: the limit is applied only after
    reranking, not during the source merge."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.01},
                "1": {"type": "noul", "noul": 0.99},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, limit=1)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    # both candidates, including the one beyond the output limit, are sent
    # to the decision model
    assert set(payload["questions"]) == {"0", "1"}
    # the endorsed candidate 1 is promoted into the top-1 output
    assert [int(s.subject_id) for s in result[0]] == [1]


def test_dm_rerank_blend_pure_source(app_project):
    """With model-strength=0 and the default negative gate threshold the
    blend exactly reproduces the pure source order (the floor of the
    search space); the scores are still mapped through the squash."""
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
    # the gate is exactly 1.0 at the default threshold -1.0 and the odds
    # term is a no-op with model-strength 0, so the raw scores are
    # exactly the source scores, mapped through the squash
    assert by_id[0] == pytest.approx(_squash(0.9))
    assert by_id[1] == pytest.approx(_squash(0.8))
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_dm_rerank_blend_pure_source_preserves_closely_spaced_order(app_project):
    """Regression: the pure-source point (model-strength=0,
    gate-threshold=-1) must not let the model answer change the ranking,
    even for closely spaced source scores and opposite noul scores."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    suggestions = [
        SubjectSuggestion(0, 0.5),
        SubjectSuggestion(1, 0.499999),
    ]
    # the worst-case contrast: the lower source score has the highest
    # noul score and vice versa
    noul_scores = {0: 0.0, 1: 1.0}
    blended = dm_rerank_type._blend(suggestions, noul_scores, 0.0, -1.0)
    by_id = {s.subject_id: s.score for s in blended}
    # the gate is exactly off and the odds term is a no-op, so the
    # scores are exactly the squashed source scores
    assert by_id[0] == pytest.approx(_squash(0.5), abs=0.0)
    assert by_id[1] == pytest.approx(_squash(0.499999), abs=0.0)
    # the source order is preserved exactly
    assert [s.subject_id for s in blended] == [0, 1]


def test_dm_rerank_suggest_pure_source_preserves_order(app_project):
    """Regression: through the full suggest pipeline (including the
    float32 conversion of the suggestion batch) the pure-source point
    still preserves closely spaced source scores in their source order,
    regardless of the noul scores."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.0},
                "1": {"type": "noul", "noul": 1.0},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(
            app_project, **{"model-strength": 0.0, "gate-threshold": -1.0}
        )
        with _mock_source(app_project, [(0, 0.5), (1, 0.499999)]):
            result = dm_rerank.suggest([Document(text="test document")])

    suggestions = list(result[0])
    by_id = {int(s.subject_id): s.score for s in suggestions}
    assert by_id[0] == pytest.approx(_squash(0.5))
    assert by_id[1] == pytest.approx(_squash(0.499999))
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


def test_dm_rerank_modification_time(app_project):
    """The backend has no model files of its own, so it reports the
    source project's modification time, or None when the source does
    not have one."""
    dm_rerank = _make_backend(app_project)
    mtime = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    with _mock_source(app_project, [], modification_time=mtime):
        assert dm_rerank.modification_time == mtime
    with _mock_source(app_project, [], modification_time=None):
        assert dm_rerank.modification_time is None


def test_dm_rerank_label_project_language(app_project):
    """Label selection prefers a nonempty project-language label, then the
    first nonempty label in any language; a subject whose labels are all
    missing, and a deprecated subject (labels is None), yield None so the
    candidate is skipped."""
    subjects = SubjectIndexFile()
    subjects.append(Subject(uri="u0", labels={"en": "Alpha"}, notation=None))
    subjects.append(Subject(uri="u1", labels={}, notation="42.42"))
    subjects.append(
        Subject(uri="u2", labels={"fi": "beta-fi", "en": "Beta"}, notation=None)
    )
    # a deprecated subject has labels set to None
    subjects.append(Subject(uri="u3", labels=None, notation=None))
    # a subject with no labels and no notation yields no label
    subjects.append(Subject(uri="u4", labels={}, notation=None))
    # CSV vocabularies store missing translations as None
    subjects.append(
        Subject(uri="u5", labels={"fi": None, "en": "Gamma"}, notation=None)
    )
    # the project-language entry and the first entry are both missing
    subjects.append(
        Subject(uri="u6", labels={"fi": None, "en": None, "sv": "Delta"}, notation=None)
    )
    # all translations missing -> no usable label
    subjects.append(Subject(uri="u7", labels={"fi": None, "en": None}, notation=None))

    proj = unittest.mock.Mock()
    proj.language = "fi"
    proj.datadir = "/tmp"
    proj.subjects = subjects
    dm_rerank = annif.backend.get_backend("dm_rerank")(
        backend_id="dm_rerank", config_params={"sources": "dummy-en"}, project=proj
    )

    # subject 2 has a fi label -> use it
    assert dm_rerank._label_for_subject(2) == "beta-fi"
    # subject 0 only has en -> fall back to that
    assert dm_rerank._label_for_subject(0) == "Alpha"
    # subject 1 has no labels -> None (candidate skipped)
    assert dm_rerank._label_for_subject(1) is None
    # subject 3 is deprecated -> None (candidate skipped)
    assert dm_rerank._label_for_subject(3) is None
    # subject 4 has neither labels nor notation -> None (candidate skipped)
    assert dm_rerank._label_for_subject(4) is None
    # subject 5: fi label missing, en label usable -> use it
    assert dm_rerank._label_for_subject(5) == "Gamma"
    # subject 6: preferred and first entries missing -> first nonempty one
    assert dm_rerank._label_for_subject(6) == "Delta"
    # subject 7: all translations missing -> None (candidate skipped)
    assert dm_rerank._label_for_subject(7) is None


def test_dm_rerank_deprecated_subjects_skipped(app_project):
    """Deprecated subjects (no labels) are skipped by the reranking service
    query: when all candidates are deprecated no request is sent at all,
    and when only some are, they are excluded from the request payload."""
    subjects = SubjectIndexFile()
    subjects.append(Subject(uri="u0", labels={"fi": "dummy"}, notation=None))
    # a deprecated subject has labels set to None
    subjects.append(Subject(uri="u1", labels=None, notation=None))

    proj = unittest.mock.Mock()
    proj.language = "fi"
    proj.datadir = "/tmp"
    proj.subjects = subjects

    dm_rerank = _make_backend(proj)
    doc = Document(text="test document")

    # all candidates deprecated -> no questions, and therefore no request
    with unittest.mock.patch("requests.post") as mock_request:
        suggestions = [SubjectSuggestion(1, 0.9)]
        result = dm_rerank._query_dm_rerank(doc, suggestions, dm_rerank.params)
        assert result == {}
        mock_request.assert_not_called()

    # mixed candidates -> the deprecated one is not sent to the service
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response
        suggestions = [SubjectSuggestion(0, 0.9), SubjectSuggestion(1, 0.8)]
        result = dm_rerank._query_dm_rerank(doc, suggestions, dm_rerank.params)

    assert result == {0: 0.9}
    payload = mock_request.call_args.kwargs["json"]
    assert set(payload["questions"]) == {"0"}


def test_dm_rerank_hyperopt(app_project):
    """The hyperparameter optimizer queries the decision model once per
    document and searches the model-strength / gate-threshold blend
    parameters over the cached scores. With max-candidates set, only the
    top candidates are sent to the decision model in _prepare."""
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

        dm_rerank = _make_backend(app_project, **{"max-candidates": 1})
        optimizer = dm_rerank.get_hp_optimizer(corpus, metric="NDCG")
        recommendation = optimizer.optimize(n_trials=3, n_jobs=1, results_file=None)

    # the decision model is queried once per document in the corpus
    assert mock_request.call_count == 2
    # with max-candidates=1 only the top candidate is sent to the model
    payload = mock_request.call_args.kwargs["json"]
    assert set(payload["questions"]) == {"0"}
    # the recommendation contains both blend parameters and a score
    assert "model-strength=" in recommendation.lines[0]
    assert "gate-threshold=" in recommendation.lines[1]
    assert recommendation.score is not None


def test_dm_rerank_hyperopt_pool_not_truncated(app_project):
    """The hyperopt optimization pool is not truncated to the output limit:
    candidates beyond the limit are also sent to the decision model in
    _prepare, so the searched blend parameters reflect the production
    behavior."""
    from annif.corpus import DocumentList

    corpus = DocumentList([Document(text="a test document")])

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
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, limit=1)
        optimizer = dm_rerank.get_hp_optimizer(corpus, metric="NDCG")
        optimizer.optimize(n_trials=2, n_jobs=1, results_file=None)

    # the candidate beyond the output limit is also queried
    payload = mock_request.call_args.kwargs["json"]
    assert set(payload["questions"]) == {"0", "1"}


def test_dm_rerank_hyperopt_enqueues_source_baseline(app_project):
    """The pure-source configuration (model-strength=0.0,
    gate-threshold=-1.0) is enqueued as a fixed baseline trial, in
    addition to the sampled trials."""
    import optuna

    corpus = unittest.mock.Mock()
    dm_rerank = _make_backend(app_project)
    optimizer = dm_rerank.get_hp_optimizer(corpus, metric="NDCG")

    study = optuna.create_study(direction="maximize")
    optimizer._enqueue_trials(study)
    assert len(study.get_trials(states=(optuna.trial.TrialState.WAITING,))) == 1
    (trial,) = study.get_trials(states=(optuna.trial.TrialState.WAITING,))
    # enqueued (fixed) trials carry their parameters in a system attribute
    assert trial.system_attrs["fixed_params"] == {
        "model-strength": 0.0,
        "gate-threshold": -1.0,
    }


def test_dm_rerank_hyperopt_recovers_source_ranking(app_project):
    """When the noul scores carry no signal (here they actively invert
    the source order) the optimizer still recommends the pure-source
    configuration, because it is evaluated as an enqueued baseline
    trial: the recommendation must never be worse than the source
    ranking."""
    from annif.corpus import DocumentList

    # gold labels match the source order exactly, so the pure-source
    # configuration achieves NDCG 1.0, while any configuration with a
    # positive model strength or an active gate swaps the two candidates
    # and scores 0.5
    corpus = DocumentList(
        [Document(text="a test document", subject_set=SubjectSet([0]))]
    )

    def mock_suggest(documents, params=None):
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
        # the decision model is confidently wrong: it inverts the source order
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.1},
                "1": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        optimizer = dm_rerank.get_hp_optimizer(corpus, metric="NDCG")
        recommendation = optimizer.optimize(n_trials=1, n_jobs=1, results_file=None)

    # the baseline trial is evaluated in addition to the sampled trial
    assert recommendation.lines == [
        "model-strength=0.0000",
        "gate-threshold=-1.0000",
    ]
    # the pure-source configuration recovers the source ranking
    assert recommendation.score == 1.0
