"""Unit tests for Annif REST API / OpenAPI spec"""

import pytest
import schemathesis
from hypothesis import settings
from hypothesis import strategies as st

import annif

cxapp = annif.create_app(config_name="annif.default_config.TestingConfig")
schema = schemathesis.openapi.from_asgi("/v1/openapi.json", app=cxapp)
schema.config.checks.positive_data_acceptance.enabled = False
schema.config.generation.allow_extra_parameters = False


INT32_MAX = 2147483647


@schemathesis.hook("filter_case")
def filter_case_limit(context, case):
    # Exclude cases where limit exceeds int32 max, since Connexion does not
    # enforce int32 range bounds from format: int32 alone.
    limit = None
    if case.query is not None and "limit" in case.query:
        limit = case.query["limit"]
    elif isinstance(case.body, dict) and "limit" in case.body:
        limit = case.body["limit"]
    if limit is not None:
        try:
            if int(limit) > INT32_MAX:
                return False
        except (TypeError, ValueError):
            pass
    return True


@schemathesis.hook("filter_body")
def filter_body(context, body):
    # Exclude body containing non-utf8 content to avoid crashing Connexion:
    # https://github.com/spec-first/connexion/issues/1860
    if body is None or isinstance(body, (dict, list, str, int, float, bool)):
        return True
    elif isinstance(body, (bytes, bytearray)):
        try:
            _ = body.decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


@schemathesis.hook("filter_path_parameters")
def filter_path_parameters(context, path_parameters):
    # Exclude path parameters containing newline to avoid crashing Connexion:
    # https://github.com/spec-first/connexion/issues/1908
    if path_parameters is not None and "project_id" in path_parameters:
        return "%0A" not in path_parameters["project_id"]
    return True


# Whitelist of project IDs that are valid for OpenAPI fuzzy testing
# Only projects that work without training (dummy backend) are included
PROJECTS_TO_TEST = (
    "dummy-fi",
    "dummy-en",
)


@schemathesis.hook("before_generate_path_parameters")
def before_generate_path_parameters(context, strategy):
    """Replace the path parameter generation strategy with a whitelist."""
    if context.operation and "project_id" in context.operation.path:
        return st.fixed_dictionaries({"project_id": st.sampled_from(PROJECTS_TO_TEST)})
    return strategy


@schema.parametrize()
@settings(max_examples=10)
def test_openapi_fuzzy(case):
    case.call_and_validate()


@pytest.mark.slow
@schema.include(path_regex="projects/{project_id}").parametrize()
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


def test_rest_suggest_payload_exceeds_max_content_length(app_client):
    # Create a payload that exceeds the MAX_CONTENT_LENGTH limit
    large_text = "A" * 3_000
    data = {"text": large_text}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest",
        data=data,
    )
    assert req.status_code == 413  # Request Entity Too Large


def test_rest_suggest_batch_payload_exceeds_max_content_length(app_client):
    # Create a payload that exceeds the MAX_CONTENT_LENGTH limit
    large_text = "A" * 3_000
    data = {"documents": [{"text": large_text}]}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest-batch",
        json=data,
    )
    assert req.status_code == 413  # Request Entity Too Large


def test_rest_suggest_payload_within_max_content_length(app_client):
    # Create a payload well within the limit
    moderate_text = "A" * 500
    data = {"text": moderate_text}
    req = app_client.post(
        "http://localhost:8000/v1/projects/dummy-fi/suggest",
        data=data,
    )
    assert req.status_code == 200
    assert "results" in req.json()


def test_rest_detect_language_payload_exceeds_max_content_length(app_client):
    # Create a payload that exceeds the MAX_CONTENT_LENGTH limit
    large_text = "A" * 3_000
    data = {"text": large_text, "languages": ["en", "fi"]}
    req = app_client.post(
        "http://localhost:8000/v1/detect-language",
        json=data,
    )
    assert req.status_code == 413  # Request Entity Too Large


def test_rest_detect_language_payload_within_max_content_length(app_client):
    small_text = "A" * 500
    data = {"text": small_text, "languages": ["en", "fi"]}
    req = app_client.post(
        "http://localhost:8000/v1/detect-language",
        json=data,
    )
    assert req.status_code == 200
    assert "results" in req.json()
