"""Unit tests for the CLM backend in Annif"""

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


def test_clm_rerank_keeps_all(app_project):
    """In rerank mode, no candidate is dropped regardless of the noul score."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.9},
                "1": {"type": "noul", "noul": 0.1},
            }
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project, mode="rerank")
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = clm.suggest([Document(text="test document")])

    suggestions = list(result[0])
    # both candidates survive
    assert len(suggestions) == 2
    # scores are rescaled by the noul score
    by_id = {int(s.subject_id): s.score for s in suggestions}
    assert by_id[0] == pytest.approx(0.9 * 0.9)
    assert by_id[1] == pytest.approx(0.8 * 0.1)
    # the reranked order is by the new score
    assert [int(s.subject_id) for s in suggestions] == [0, 1]


def test_clm_rerank_reorders(app_project):
    """A low source score with a high noul score can move above a
    higher source score with a low noul score."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {
                "0": {"type": "noul", "noul": 0.2},
                "1": {"type": "noul", "noul": 0.9},
            }
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project, mode="rerank")
        # subject 0 has the higher source score, subject 1 the lower
        with _mock_source(app_project, [(0, 0.9), (1, 0.8)]):
            result = clm.suggest([Document(text="test document")])

    suggestions = list(result[0])
    # 0.9*0.2=0.18 vs 0.8*0.9=0.72 -> subject 1 comes first
    assert [int(s.subject_id) for s in suggestions] == [1, 0]
    assert suggestions[0].score == pytest.approx(0.72)
    assert suggestions[1].score == pytest.approx(0.18)


def test_clm_rerank_threshold_ignored(app_project):
    """In rerank mode the threshold parameter does not drop candidates."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.1}}
        }
        mock_request.return_value = mock_response

        clm = _make_backend(app_project, mode="rerank", threshold=0.6)
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    suggestions = list(result[0])
    assert len(suggestions) == 1
    assert suggestions[0].score == pytest.approx(0.09)


def test_clm_suggest_request_error(app_project):
    """A failed CLM request raises OperationFailedException after all
    retries have been exhausted."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        clm = _make_backend(app_project, retries=1)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                clm.suggest([Document(text="test document")])

        # 1 initial attempt + 1 retry
        assert mock_request.call_count == 2


def test_clm_suggest_retries_on_failure(app_project):
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

        clm = _make_backend(app_project)
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    assert mock_request.call_count == 2
    mock_sleep.assert_called_once()
    suggestions = list(result[0])
    assert [int(s.subject_id) for s in suggestions] == [0]


def test_clm_suggest_no_retries(app_project):
    """With retries=0 a single failure raises immediately."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        unittest.mock.patch("time.sleep"),
    ):
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )

        clm = _make_backend(app_project, retries=0)
        with _mock_source(app_project, [(0, 0.9)]):
            with pytest.raises(OperationFailedException):
                clm.suggest([Document(text="test document")])

    assert mock_request.call_count == 1


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


def test_clm_custom_instruction(app_project):
    """A custom instruction template replaces the default proposition."""
    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response

        clm = _make_backend(
            app_project,
            instruction="Does this document relate to {label}?",
        )
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["questions"]["0"]["instructions"] == (
        "Does this document relate to dummy?"
    )
    assert [int(s.subject_id) for s in result[0]] == [0]


def test_clm_instruction_missing_placeholder(app_project):
    """An instruction without the {label} placeholder is rejected."""
    with (
        unittest.mock.patch("requests.post") as mock_request,
        _mock_source(app_project, [(0, 0.9)]),
    ):
        clm = _make_backend(app_project, instruction="No label here")
        with pytest.raises(ConfigurationException):
            clm.suggest([Document(text="test document")])

    mock_request.assert_not_called()


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


def _mock_project(app_project, language):
    """A mock project using the real dummy subject index"""
    proj = unittest.mock.Mock()
    proj.language = language
    proj.subjects = app_project.subjects
    return proj


def _clm_with_graph(project, graph, **params):
    """Build a CLM backend with the given rdflib graph (or list of triples)
    as its vocab graph."""
    import rdflib

    if isinstance(graph, list):
        rdflib_graph = rdflib.Graph()
        for s, p, o in graph:
            rdflib_graph.add((rdflib.URIRef(s), p, o))
        graph = rdflib_graph
    clm_type = annif.backend.get_backend("clm")
    base = {"sources": "dummy-en"}
    base.update(params)
    clm = clm_type(backend_id="clm", config_params=base, project=project)
    clm._graph = graph
    return clm


def test_clm_info_empty_by_default(app_project):
    """With the default info parameter no subject info is requested."""
    clm = _make_backend(app_project)
    # no vocab graph should be loaded at all
    assert clm._graph is None
    assert clm._subject_info(0, clm.params) == ""


def test_clm_info_definition_and_scope_note(app_project):
    """skos:definition and skos:scopeNote are picked up, project language
    preferred."""
    import rdflib
    from rdflib.namespace import SKOS

    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        [
            (
                "http://example.org/dummy",
                SKOS.definition,
                rdflib.Literal("An English definition", "en"),
            ),
            (
                "http://example.org/dummy",
                SKOS.definition,
                rdflib.Literal("Suomen maärítelmä", "fi"),
            ),
            (
                "http://example.org/dummy",
                SKOS.scopeNote,
                rdflib.Literal("Scope note", "en"),
            ),
        ],
        info="definition,scopeNote",
    )
    # fi definition comes first, then the en one, then the scope note
    info = clm._subject_info(0, {"info": "definition,scopeNote", "max-info-len": 300})
    assert info == "Suomen maärítelmä An English definition Scope note"


def test_clm_info_alt_label_and_broader(app_project):
    """altLabel and broader (rendered via the subject index) are included."""
    import rdflib
    from rdflib.namespace import SKOS

    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        [
            (
                "http://example.org/dummy",
                SKOS.altLabel,
                rdflib.Literal("alt-fi", "fi"),
            ),
            (
                "http://example.org/dummy",
                SKOS.broader,
                rdflib.URIRef("http://example.org/none"),
            ),
        ],
        info="altLabel,broader",
    )
    info = clm._subject_info(0, {"info": "altLabel,broader", "max-info-len": 300})
    assert "also known as: alt-fi" in info
    assert "a subfield of: none-fi" in info


def test_clm_info_truncated(app_project):
    """Info is truncated to max-info-len characters."""
    import rdflib
    from rdflib.namespace import SKOS

    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        [
            (
                "http://example.org/dummy",
                SKOS.definition,
                rdflib.Literal("x" * 500, "fi"),
            )
        ],
        info="definition",
    )
    info = clm._subject_info(0, {"info": "definition", "max-info-len": 10})
    assert len(info) == 10


def test_clm_info_note(app_project):
    """skos:note values are picked up, project language preferred."""
    import rdflib
    from rdflib.namespace import SKOS

    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        [
            (
                "http://example.org/dummy",
                SKOS.note,
                rdflib.Literal("A note in English", "en"),
            ),
            (
                "http://example.org/dummy",
                SKOS.note,
                rdflib.Literal("Huomautus suomeksi", "fi"),
            ),
        ],
        info="note",
    )
    info = clm._subject_info(0, {"info": "note", "max-info-len": 300})
    assert info == "Huomautus suomeksi A note in English"


def test_clm_info_collection(app_project):
    """Collections whose member the concept is are included via their
    labels, falling back to other languages and plain rdfs:label."""
    import rdflib
    from rdflib.namespace import RDFS, SKOS

    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        [
            (
                "http://example.org/dummy",
                SKOS.prefLabel,
                rdflib.Literal("dummy", "fi"),
            ),
            # fi prefLabel
            (
                "http://example.org/collection1",
                SKOS.prefLabel,
                rdflib.Literal("Kokoelma yksi", "fi"),
            ),
            # only an sv prefLabel -> falls back to it
            (
                "http://example.org/collection2",
                SKOS.prefLabel,
                rdflib.Literal("Samling två", "sv"),
            ),
            # only a plain rdfs:label -> used as a last resort
            (
                "http://example.org/collection3",
                RDFS.label,
                rdflib.Literal("Kokoelma kolme", "fi"),
            ),
            (
                "http://example.org/collection1",
                SKOS.member,
                rdflib.URIRef("http://example.org/dummy"),
            ),
            (
                "http://example.org/collection2",
                SKOS.member,
                rdflib.URIRef("http://example.org/dummy"),
            ),
            (
                "http://example.org/collection3",
                SKOS.member,
                rdflib.URIRef("http://example.org/dummy"),
            ),
        ],
        info="collection",
    )
    info = clm._subject_info(0, {"info": "collection", "max-info-len": 300})
    assert "member of: Kokoelma yksi, Samling två, Kokoelma kolme" in info


def _preflabel_graph():
    import rdflib
    from rdflib.namespace import SKOS

    graph = rdflib.Graph()
    uri = rdflib.URIRef("http://example.org/dummy")
    graph.add((uri, SKOS.prefLabel, rdflib.Literal("dummy-fi", "fi")))
    graph.add((uri, SKOS.prefLabel, rdflib.Literal("dummy-en", "en")))
    graph.add((uri, SKOS.prefLabel, rdflib.Literal("dummy-sv", "sv")))
    graph.add((uri, SKOS.prefLabel, rdflib.Literal("no-lang")))
    return graph


def test_clm_info_preflabel_all_except_project(app_project):
    """A bare prefLabel selects every language except the project language,
    tagged by language."""
    clm = _clm_with_graph(
        _mock_project(app_project, "fi"), _preflabel_graph(), info="prefLabel"
    )
    info = clm._subject_info(0, {"info": "prefLabel", "max-info-len": 300})
    assert "dummy-fi" not in info
    assert "en: dummy-en" in info
    assert "sv: dummy-sv" in info
    # a label without a language tag is included untagged
    assert "no-lang" in info
    # exactly the three non-fi labels, nothing else
    assert info.count("dummy") == 2


def test_clm_info_preflabel_single_language(app_project):
    """prefLabel(en) selects only the given language."""
    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        _preflabel_graph(),
        info="prefLabel(en)",
    )
    info = clm._subject_info(0, {"info": "prefLabel(en)", "max-info-len": 300})
    assert info == "en: dummy-en"


def test_clm_info_preflabel_language_list(app_project):
    """prefLabel(en,sv) selects the given languages, in sorted order."""
    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        _preflabel_graph(),
        info="prefLabel(en,sv)",
    )
    info = clm._subject_info(0, {"info": "prefLabel(en,sv)", "max-info-len": 300})
    assert info == "en: dummy-en sv: dummy-sv"


def test_clm_info_unknown_type(app_project):
    """An unknown info type raises ConfigurationException."""
    import rdflib
    from rdflib.namespace import SKOS

    clm = _clm_with_graph(
        _mock_project(app_project, "fi"),
        [
            (
                "http://example.org/dummy",
                SKOS.definition,
                rdflib.Literal("x", "fi"),
            )
        ],
        info="bogus",
    )
    with pytest.raises(ConfigurationException):
        clm._subject_info(0, {"info": "bogus", "max-info-len": 10})


def test_clm_instruction_with_info(app_project):
    """A {info} placeholder in the instruction is filled from the graph."""
    import rdflib
    from rdflib.namespace import SKOS

    graph = rdflib.Graph()
    uri = rdflib.URIRef("http://example.org/dummy")
    graph.add((uri, SKOS.definition, rdflib.Literal("A dummy concept", "en")))

    clm = _make_backend(
        app_project,
        instruction="{label}: {info}",
        info="definition",
    )
    clm._graph = graph

    with unittest.mock.patch("requests.post") as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "answers": {"0": {"type": "noul", "noul": 0.9}}
        }
        mock_request.return_value = mock_response
        with _mock_source(app_project, [(0, 0.9)]):
            result = clm.suggest([Document(text="test document")])

    payload = mock_request.call_args.kwargs["json"]
    assert payload["questions"]["0"]["instructions"] == ("dummy: A dummy concept")
    assert [int(s.subject_id) for s in result[0]] == [0]


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
