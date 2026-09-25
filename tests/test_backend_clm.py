"""Unit tests for the CLM backend in Annif"""

import unittest.mock

import pytest
import requests.exceptions

import annif.backend
from annif.corpus import Document
from annif.exception import NotSupportedException, OperationFailedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.vocab import Subject


def _make_backend(project, **params):
    base = {
        "sources": "dummy-en",
        "endpoint": "http://127.0.0.1:8700",
        "threshold": 0.6,
    }
    base.update(params)
    clm_type = annif.backend.get_backend("clm")
    return clm_type(backend_id="clm", config_params=base, project=project)


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


def test_clm_suggest_keeps_valid(app_project):
    """Candidates with noul >= threshold are kept at their merged score."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.7},
            }
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = clm.suggest([Document(text="test document")])

    suggestions = list(result[0])
    # scores survive the round trip as float32
    assert [int(s.subject_id) for s in suggestions] == [0, 1]
    assert suggestions[0].score == pytest.approx(0.9)
    assert suggestions[1].score == pytest.approx(0.8)

    # verify the request payload shape sent to the CLM service
    payload = mock_request.call_args.kwargs["json"]
    assert payload["state"] == "test document"
    assert payload["model"] == "clm-latest"
    assert payload["questions"]["0"]["instructions"] == "This document is about dummy."
    assert payload["questions"]["1"]["instructions"] == "This document is about none."
    # the endpoint parameter is used as the base URL for /v1/systemone
    assert mock_request.call_args.args[0] == "http://127.0.0.1:8700/v1/systemone"


def test_clm_suggest_drops_invalid(app_project):
    """Candidates with noul below threshold are dropped."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.3},
            }
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = clm.suggest([Document(text="test document")])

    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] == [0]
    # the kept candidate retains its source score (float32 precision)
    assert suggestions[0].score == pytest.approx(0.9)


def test_clm_suggest_threshold_boundary(app_project):
    """A candidate exactly at the threshold is kept (>= semantics)."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.6}}
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    assert [s.subject_id for s in result[0]] == [0]


def test_clm_suggest_custom_threshold(app_project):
    """A custom threshold higher than the score drops the candidate."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project, threshold=0.95)
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    assert [s.subject_id for s in result[0]] == []


def test_clm_suggest_all_dropped(app_project):
    """If no candidate passes the threshold, an empty result is returned."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.1}}
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    assert list(result[0]) == []


def test_clm_suggest_request_error(app_project):
    """A failed CLM request raises OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                clm.suggest([Document(text="test document")])


def test_clm_suggest_json_error(app_project):
    """A non-JSON CLM response raises OperationFailedException."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.side_effect = ValueError("not json")
        mock_request.return_value = mock_response

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                clm.suggest([Document(text="test document")])


def test_clm_train_not_supported(app_project):
    """Training the CLM backend raises NotSupportedException."""
    clm = _make_backend(app_project)
    corpus = unittest.mock.Mock()
    with pytest.raises(NotSupportedException):
        clm.train(corpus)


def test_clm_is_trained(app_project):
    """is_trained is True only when every source is trained."""
    clm = _make_backend(app_project)
    with _mock_source(app_project, [], is_trained=True):
        assert clm.is_trained is True
    with _mock_source(app_project, [], is_trained=False):
        assert clm.is_trained is False


def test_clm_label_project_language(app_project):
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
    clm = annif.backend.get_backend("clm")(
        backend_id="clm", config_params={"sources": "dummy-en"}, project=proj
    )

    # subject 2 has a fi label -> use it
    assert clm._label_for_subject(2) == "beta-fi"
    # subject 0 only has en -> fall back to that
    assert clm._label_for_subject(0) == "Alpha"
    # subject 1 has no labels -> fall back to notation
    assert clm._label_for_subject(1) == "42.42"
    # subject 3 has neither -> None (candidate skipped)
    assert clm._label_for_subject(3) is None
