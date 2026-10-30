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


def test_dm_rerank_input_rules_prefix(app_project):
    """With input-rules enabled the built-in indexing-policy rules are
    prepended to the document input."""
    from annif.backend.dm_rerank import INPUT_RULES

    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"input-rules": "true"})
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == INPUT_RULES + "test document"


def test_dm_rerank_input_prefix(app_project):
    """A custom input-prefix is prepended to the document input."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"input-prefix": "Custom rule. "})
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "Custom rule. test document"


def test_dm_rerank_input_prefix_overrides_input_rules(app_project):
    """input-prefix takes precedence when input-rules is also enabled."""
    from annif.backend.dm_rerank import INPUT_RULES

    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(
            app_project, **{"input-rules": "true", "input-prefix": "Custom rule. "}
        )
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "Custom rule. test document"
    assert not payload["state"].startswith(INPUT_RULES)


def test_dm_rerank_input_transform(app_project):
    """The input-transform parameter is applied to the document text
    before it is sent to the reranking service."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"input-transform": "limit(10)"})
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    # the input is truncated to 10 characters
    assert payload["state"] == "test docum"


def test_dm_rerank_input_transform_gets_project(app_project):
    """The backend's project is passed to the input transform, so that
    project-dependent transforms such as filter_lang can use its language
    and analyzer."""
    dm_rerank = _make_backend(app_project, **{"input-transform": "pass"})
    transform = dm_rerank._get_input_transform(dm_rerank.params)
    assert transform.project is app_project


def test_dm_rerank_input_transform_per_call_override(app_project):
    """A per-call input-transform override is honored, even when the
    configured parameter is pass (the default)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest(
                [Document(text="test document")], {"input-transform": "limit(10)"}
            )

    payload = mock_request.call_args.kwargs["json"]
    # the override truncates the input to 10 characters
    assert payload["state"] == "test docum"


def test_dm_rerank_input_transform_override_does_not_poison_cache(app_project):
    """A per-call override does not affect subsequent calls, which keep
    using the configured transform."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest(
                [Document(text="test document")], {"input-transform": "limit(10)"}
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
    only with a malformed decision model score from the service) are clamped to
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
    source * odds(p)^model-strength * gate, where the odds term is
    (p + eps) / (1 - p + eps) and the gate is a sigmoid on the raw
    decision model score."""
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
    """The decision model score can reorder candidates: a low source score
    with a high decision model score moves above a high source score with a
    low decision model score."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    suggestions = [
        SubjectSuggestion(0, 0.9),
        SubjectSuggestion(1, 0.8),
        SubjectSuggestion(2, 0.7),
    ]
    dm_scores = {0: 0.01, 1: 0.99, 2: 0.5}
    blended = dm_rerank_type._blend(suggestions, dm_scores, 1.0, -0.25)
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
    dm_scores = {0: 0.9, 1: 0.1}
    blended = dm_rerank_type._blend(suggestions, dm_scores, 1.5, -1.0)
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
    """With model-strength=0 and a positive gate threshold the decision model score
    acts as a precision gate: a low-decision-model-score candidate is demoted below a
    high-decision-model-score one despite the better source score."""
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


@pytest.mark.parametrize("threshold", [100, 2.0, -2.0, float("nan"), float("inf")])
def test_dm_rerank_gate_threshold_out_of_range(app_project, threshold):
    """A gate-threshold outside [-1.0, 1.0] (including non-finite
    values) is rejected: decision model scores are in [0, 1], so such a threshold
    is meaningless, and a sufficiently large positive one would overflow
    the sigmoid's math.exp instead of producing a near-zero gate."""
    dm_rerank = _make_backend(app_project, **{"gate-threshold": threshold})
    with _mock_source(app_project, [(0, 0.9)]):
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest([Document(text="test document")])


@pytest.mark.parametrize("threshold", ["not-a-number", None])
def test_dm_rerank_gate_threshold_not_a_number(app_project, threshold):
    """A gate-threshold that cannot be parsed as a number is rejected
    with a configuration error rather than a raw TypeError."""
    dm_rerank = _make_backend(app_project, **{"gate-threshold": threshold})
    with _mock_source(app_project, [(0, 0.9)]):
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_gate_threshold_override_out_of_range(app_project):
    """Per-call parameter overrides are validated as well: a valid
    configured threshold does not shield an out-of-range override
    passed to suggest() (the dynamic override path)."""
    dm_rerank = _make_backend(app_project, **{"gate-threshold": 0.4})
    with _mock_source(app_project, [(0, 0.9)]):
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest([Document(text="test document")], {"gate-threshold": 100})


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 0.5, 1.0])
def test_dm_rerank_gate_threshold_boundary(app_project, threshold):
    """Gate thresholds at and inside the validated boundary of
    [-1.0, 1.0] are accepted and do not overflow the sigmoid."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.1},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"gate-threshold": threshold})
        with _mock_source(app_project, [(0, 0.9), (1, 0.1)]):
            result = dm_rerank.suggest([Document(text="test document")])
    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] in ([0, 1], [1, 0])
    for suggestion in suggestions:
        assert 0.0 <= suggestion.score < 1.0


def test_dm_rerank_blend_extreme_threshold_no_overflow():
    """_blend itself stays safe for thresholds within the validated
    range even at the decision model score boundary (the pre-validation guarantees the
    sigmoid argument stays bounded)."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    suggestions = [SubjectSuggestion(0, 0.5)]
    # threshold 1.0, decision model score 0.0: the largest (in magnitude-safe) argument
    # within the validated range
    blended = dm_rerank_type._blend(suggestions, {0: 0.0}, 1.0, 1.0)
    assert len(blended) == 1
    assert 0.0 <= blended[0].score < 1.0


@pytest.mark.parametrize(
    "strength", [float("nan"), float("inf"), 1e308, 100.0, 2.1, -1.0, -100.0]
)
def test_dm_rerank_model_strength_out_of_range(app_project, strength):
    """A model-strength outside [0.0, 2.0] (including non-finite values)
    is rejected with a configuration error. A sufficiently large finite
    value (e.g. 1e308 with a decision model score of 1.0) would otherwise raise
    OverflowError in the odds exponentiation, and a negative value would
    invert the model's verdict."""
    dm_rerank = _make_backend(app_project, **{"model-strength": strength})
    with _mock_source(app_project, [(0, 0.9)]):
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest([Document(text="test document")])


@pytest.mark.parametrize("strength", ["not-a-number", None])
def test_dm_rerank_model_strength_not_a_number(app_project, strength):
    """A model-strength that cannot be parsed as a number is rejected
    with a configuration error rather than a raw TypeError."""
    dm_rerank = _make_backend(app_project, **{"model-strength": strength})
    with _mock_source(app_project, [(0, 0.9)]):
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_model_strength_override_out_of_range(app_project):
    """Per-call parameter overrides of model-strength are validated as
    well, not just the configured value."""
    dm_rerank = _make_backend(app_project, **{"model-strength": 0.3})
    with _mock_source(app_project, [(0, 0.9)]):
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest(
                [Document(text="test document")], {"model-strength": float("nan")}
            )
        with pytest.raises(ConfigurationException):
            dm_rerank.suggest(
                [Document(text="test document")], {"model-strength": 1e308}
            )


@pytest.mark.parametrize("strength", [0.0, 0.3, 1.45, 2.0])
def test_dm_rerank_model_strength_in_range(app_project, strength):
    """Model strengths at and inside the validated boundary of [0.0, 2.0]
    are accepted and do not overflow the odds exponentiation."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 1.0},
                "1": {"type": "noul", "noul": 0.1},
            }
        }
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project, **{"model-strength": strength})
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = dm_rerank.suggest([Document(text="test document")])
    suggestions = list(result[0])
    for suggestion in suggestions:
        assert 0.0 <= suggestion.score < 1.0
        assert math.isfinite(suggestion.score)


def test_dm_rerank_blend_missing_score(app_project):
    """A candidate without a decision model score gets its source score on the same
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
    # subject 1 has no decision model score: its raw source score is mapped through
    # the same squash so both candidates are on one scale; the gate is
    # exactly off at the default threshold -1.0
    assert by_id[1] == pytest.approx(_squash(0.8))
    assert by_id[0] == pytest.approx(_squash(0.9 * _odds(0.9) ** 0.3))
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_dm_rerank_blend_no_scores(app_project):
    """When no candidate has a decision model score (empty answers), every score is
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
    even for closely spaced source scores and opposite decision model scores."""
    dm_rerank_type = annif.backend.get_backend("dm_rerank")
    suggestions = [
        SubjectSuggestion(0, 0.5),
        SubjectSuggestion(1, 0.499999),
    ]
    # the worst-case contrast: the lower source score has the highest
    # decision model score and vice versa
    dm_scores = {0: 0.0, 1: 1.0}
    blended = dm_rerank_type._blend(suggestions, dm_scores, 0.0, -1.0)
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
    regardless of the decision model scores."""
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


def _http_error(status_code):
    """Return an HTTPError for a deterministic non-success response."""
    response = unittest.mock.Mock()
    response.status_code = status_code
    return requests.exceptions.HTTPError(f"{status_code} error", response=response)


def test_dm_rerank_suggest_no_retry_on_4xx(app_project):
    """A non-retryable 4xx (deterministic client error, here 404) fails
    fast without any retry."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep") as mock_sleep,
    ):
        mock_request.side_effect = _http_error(404)

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])

    assert mock_request.call_count == 1
    mock_sleep.assert_not_called()


def _ok_response(score: float = 0.9):
    """Return a successful reranking response for subject 0."""
    mock_response = unittest.mock.Mock()
    mock_response.json.return_value = {
        "answers": {"0": {"type": "noul", "noul": score}}
    }
    return mock_response


def test_dm_rerank_400_retries_with_reduced_input(app_project):
    """A 400 response is retried with the input reduced to 75% of the
    original (and 50% if needed); the reductions are applied to the
    original input, not cascaded."""
    long_input = "a" * 1000
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep") as mock_sleep,
    ):
        mock_request.side_effect = [
            _http_error(400),
            _http_error(400),
            _ok_response(),
        ]

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = dm_rerank.suggest([Document(text=long_input)])

    assert mock_request.call_count == 3
    # the input (without the empty prefix) must be reduced to 75% then 50%
    # of the original 1000-char input, not 50% of 75%
    inputs = [call.kwargs["json"]["state"] for call in mock_request.call_args_list]
    assert len(inputs[0]) == 1000
    assert len(inputs[1]) == 750
    assert len(inputs[2]) == 500
    # the successful (reduced) request yields the decision model score
    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] == [0]
    # no transient backoff is used for the 400 cascade
    mock_sleep.assert_not_called()


def test_dm_rerank_400_first_retry_succeeds(app_project):
    """If the 75% retry succeeds, no further reduction is attempted."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = [_http_error(400), _ok_response()]

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = dm_rerank.suggest([Document(text="a" * 1000)])

    assert mock_request.call_count == 2
    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] == [0]


def test_dm_rerank_400_all_retries_fail_keeps_source_scores(app_project):
    """If the request is still rejected (400) after the last reduction,
    the document is not aborted: no exception is raised and the document
    is scored with its plain source score."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = [_http_error(400)] * 3

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = dm_rerank.suggest([Document(text="a" * 1000)])

    assert mock_request.call_count == 3
    # the source score passes through the blend unmodified by any decision model
    # score (only the usual squash is applied)
    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] == [0]
    assert suggestions[0].score == pytest.approx(_squash(0.9))


def test_dm_rerank_400_preserves_input_prefix_on_retries(app_project):
    """The input-prefix is kept intact on the 400 retries; only the
    document input is reduced."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = [_http_error(400), _ok_response()]

        dm_rerank = _make_backend(app_project, **{"input-prefix": "RULES: "}, limit=100)
        with _mock_source(app_project, [(0, 0.9)]):
            dm_rerank.suggest([Document(text="a" * 1000)])

    inputs = [call.kwargs["json"]["state"] for call in mock_request.call_args_list]
    # the prefix is kept on the retry, only the input is reduced
    assert inputs[0] == "RULES: " + "a" * 1000
    assert inputs[1] == "RULES: " + "a" * 750


def test_dm_rerank_suggest_retry_on_429_and_5xx(app_project):
    """429 and 5xx responses are treated as transient and retried."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = [_http_error(429), _http_error(503)]

        dm_rerank = _make_backend(app_project, retries=1)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])

    assert mock_request.call_count == 2


def test_dm_rerank_suggest_backoff_starts_at_one_second(app_project):
    """The retry backoff sleeps are 1s, 2s, 4s, ... (not 2s, 4s, 8s)."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep") as mock_sleep,
    ):
        mock_request.side_effect = requests.exceptions.Timeout("timeout")

        dm_rerank = _make_backend(app_project, retries=2)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])

    assert [call.args[0] for call in mock_sleep.call_args_list] == [1, 2]


def test_dm_rerank_suggest_invalid_answers(app_project):
    """A response whose 'answers' is not an object raises
    OperationFailedException instead of an uncaught AttributeError."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": None}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_suggest_invalid_answer_entry(app_project):
    """A response whose answer entry is not an object raises
    OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {"0": "noul 0.9"}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_suggest_invalid_answer_key(app_project):
    """A response whose answer key is not a subject ID raises
    OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {"not-a-number": {"noul": 0.9}}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_suggest_nonnumeric_score(app_project):
    """A response with a nonnumeric decision model score raises
    OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {"0": {"noul": "0.9"}}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])


def test_dm_rerank_suggest_out_of_range_score(app_project):
    """A response with a decision model score outside [0, 1] raises
    OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {"answers": {"0": {"noul": 1.5}}}
        mock_request.return_value = mock_response

        dm_rerank = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                dm_rerank.suggest([Document(text="test document")])


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
        123,
        None,
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

    mock_source = unittest.mock.Mock(
        is_trained=True,
        suggest=mock_suggest,
        initialize=unittest.mock.Mock(),
    )
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch.object(
            app_project.registry, "get_project", return_value=mock_source
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

    # the source project is initialized before it is used for suggestions
    mock_source.initialize.assert_called_once()
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
    gate-threshold=-1.0) is enqueued as a fixed baseline trial,
    counting towards the requested number of trials."""
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
    """When the decision model scores carry no signal (here they actively invert
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

    # with n_trials=1 the enqueued baseline is the only trial run
    assert recommendation.lines == [
        "model-strength=0.0000",
        "gate-threshold=-1.0000",
    ]
    # the pure-source configuration recovers the source ranking
    assert recommendation.score == 1.0
