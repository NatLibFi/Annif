"""Unit tests for Annif REST API / OpenAPI spec"""

import json

import pytest
import schemathesis
from hypothesis import settings

schema = schemathesis.from_path("annif/openapi/annif.yaml")


@schemathesis.check
def check_cors(response, case):
    assert response.headers["access-control-allow-origin"] == "*"


@schema.parametrize()
@settings(max_examples=10)
def test_openapi_fuzzy(case, app):
    response = case.call_wsgi(app)
    case.validate_response(response, additional_checks=(check_cors,))


@pytest.mark.slow
@schema.parametrize(endpoint="/v1/projects/{project_id}")
@settings(max_examples=50)
def test_openapi_fuzzy_target_dummy_fi(case, app):
    case.path_parameters = {"project_id": "dummy-fi"}
    response = case.call_wsgi(app)
    case.validate_response(response)


def test_openapi_list_projects(app_client):
    req = app_client.get("http://localhost:8000/v1/projects")
    assert req.status_code == 200
    assert "projects" in req.get_json()


def test_openapi_show_project(app_client):
    req = app_client.get("http://localhost:8000/v1/projects/dummy-fi")
    assert req.status_code == 200
    assert req.get_json()["project_id"] == "dummy-fi"


def test_openapi_show_project_nonexistent(app_client):
    req = app_client.get("http://localhost:8000/v1/projects/nonexistent")
    assert req.status_code == 404


def test_openapi_suggest(app_client):
    data = {"text": "example text"}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest", data=data
    )
    assert req.status_code == 200
    assert "results" in req.get_json()


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
    body = req.get_json()
    assert len(body) == 32
    assert body[0]["results"][0]["label"] == "dummy-fi"


def test_openapi_suggest_batch_too_many_documents(app_client):
    data = {"documents": [{"text": "A quick brown fox jumped over the lazy dog."}] * 33}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest-batch", json=data
    )
    assert req.status_code == 400
    assert req.get_json()["detail"] == "too many items - 'documents'"


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


def test_openapi_reconcile_metadata(app_client):
    req = app_client.get("http://localhost:8000/v1/projects/dummy-fi/reconcile")
    assert req.status_code == 200
    assert "name" in req.get_json()


def test_openapi_reconcile_metadata_nonexistent(app_client):
    req = app_client.get("http://localhost:8000/v1/projects/nonexistent/reconcile")
    assert req.status_code == 404


def test_openapi_reconcile_metadata_queries(app_client):
    req = app_client.get(
        'http://localhost:8000/v1/projects/dummy-fi/reconcile?queries=\
         {"q0": {"query": "example text"}}'
    )
    assert req.status_code == 200
    assert "result" in req.get_json()["q0"]


def test_openapi_reconcile_metadata_queries_nonexistent(app_client):
    req = app_client.get(
        'http://localhost:8000/v1/projects/nonexistent/reconcile?queries=\
         {"q0": {"query": "example text"}}'
    )
    assert req.status_code == 404


def test_openapi_reconcile(app_client):
    data = {"queries": json.dumps({"q0": {"query": "example text"}})}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/reconcile", data=data
    )
    assert req.status_code == 200
    assert "result" in req.get_json()["q0"]


def test_openapi_reconcile_nonexistent(app_client):
    data = {"queries": json.dumps({"q0": {"query": "example text"}})}
    req = app_client.post(
        "http://localhost:8000/v1/projects/nonexistent/reconcile", data=data
    )
    assert req.status_code == 404


def test_openapi_reconcile_suggest(app_client):
    req = app_client.get(
        "http://localhost:8000/v1/projects/dummy-fi/reconcile/suggest/entity?prefix=example"
    )
    assert req.status_code == 200
    assert "result" in req.get_json()


def test_openapi_reconcile_suggest_nonexistent(app_client):
    req = app_client.get(
        "http://localhost:8000/v1/projects/nonexistent/reconcile/suggest/entity?prefix=example"
    )
    assert req.status_code == 404
