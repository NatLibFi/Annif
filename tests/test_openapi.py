"""Unit tests for Annif REST API / OpenAPI spec"""

import pytest
import schemathesis
from hypothesis import settings

import annif

cxapp = annif.create_app(config_name="annif.default_config.TestingConfig")
schema = schemathesis.from_path("annif/openapi/annif.yaml", app=cxapp)


@schemathesis.hook("filter_path_parameters")
def filter_path_parameters(context, path_parameters):
    # Exclude path parameters containing newline which crashes application
    # https://github.com/spec-first/connexion/issues/1908
    if path_parameters is not None and "project_id" in path_parameters:
        return "%0A" not in path_parameters["project_id"]
    return True


@schema.parametrize()
@settings(max_examples=10)
def test_openapi_fuzzy(case):
    case.call_and_validate()


@pytest.mark.slow
@schema.include(path_regex="/v1/projects/{project_id}").parametrize()
@settings(max_examples=50)
def test_openapi_fuzzy_target_dummy_fi(case):
    case.path_parameters = {"project_id": "dummy-fi"}
    case.call_and_validate()


def test_openapi_cors(app_client):
    # test that the service supports CORS by simulating a cross-origin request
    app_client.headers = {"Origin": "http://somedomain.com"}
    req = app_client.get(
        "http://localhost:8000/v1/projects",
    )
    assert req.headers["access-control-allow-origin"] == "*"


def test_openapi_list_projects(app_client):
    req = app_client.get("http://localhost:8000/v1/projects")
    assert req.status_code == 200
    assert "projects" in req.json()


def test_openapi_show_project(app_client):
    req = app_client.get("http://localhost:8000/v1/projects/dummy-fi")
    assert req.status_code == 200
    assert req.json()["project_id"] == "dummy-fi"


def test_openapi_show_project_nonexistent(app_client):
    req = app_client.get("http://localhost:8000/v1/projects/nonexistent")
    assert req.status_code == 404


def test_openapi_suggest(app_client):
    data = {"text": "example text"}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest", data=data
    )
    assert req.status_code == 200
    assert "results" in req.json()


def test_openapi_suggest_nonexistent(app_client):
    data = {"text": "example text"}
    req = app_client.post(
        "http://localhost:8000/v1/projects/nonexistent/suggest", data=data
    )
    assert req.status_code == 404


def test_openapi_suggest_novocab(app_client):
    data = {"text": "example text"}
    req = app_client.post(
        "http://localhost:8000/v1/projects/novocab/suggest", data=data
    )
    assert req.status_code == 503


def test_openapi_suggest_batch(app_client):
    data = {"documents": [{"text": "A quick brown fox jumped over the lazy dog."}] * 32}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest-batch", json=data
    )
    assert req.status_code == 200
    body = req.json()
    assert len(body) == 32
    assert body[0]["results"][0]["label"] == "dummy-fi"


def test_openapi_suggest_batch_too_many_documents(app_client):
    data = {"documents": [{"text": "A quick brown fox jumped over the lazy dog."}] * 33}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest-batch", json=data
    )
    assert req.status_code == 400
    assert req.json()["detail"] == "too many items - 'documents'"


def test_openapi_learn(app_client):
    data = [
        {
            "text": "the quick brown fox",
            "subjects": [{"uri": "http://example.org/fox", "label": "fox"}],
        }
    ]
    req = app_client.post("http://localhost:8000/v1/projects/dummy-fi/learn", json=data)
    assert req.status_code == 204


def test_openapi_learn_nonexistent(app_client):
    data = []
    req = app_client.post(
        "http://localhost:8000/v1/projects/nonexistent/learn", json=data
    )
    assert req.status_code == 404


def test_openapi_learn_novocab(app_client):
    data = []
    req = app_client.post("http://localhost:8000/v1/projects/novocab/learn", json=data)
    assert req.status_code == 503


def test_rest_detect_language_no_candidates(app_client):
    data = {"text": "example text", "languages": []}
    req = app_client.post("http://localhost:8000/v1/detect-language", json=data)
    assert req.status_code == 400


def test_rest_detect_language_too_many_candidates(app_client):
    data = {"text": "example text", "languages": ["en", "fr", "de", "it", "es", "nl"]}
    req = app_client.post("http://localhost:8000/v1/detect-language", json=data)
    assert req.status_code == 400
