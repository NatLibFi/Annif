"""Unit tests for Annif REST API / OpenAPI spec"""

import pytest
import schemathesis
from hypothesis import settings

schema = schemathesis.from_path("annif/openapi/annif.yaml")


@schemathesis.check
def check_cors(response, case):
    assert response.headers["access-control-allow-origin"] == "*"


@schema.parametrize()
@settings(max_examples=10)
def test_api(case, app):
    response = case.call_wsgi(app)
    case.validate_response(response, additional_checks=(check_cors,))


@pytest.mark.slow
@schema.parametrize(endpoint="/v1/projects/{project_id}")
@settings(max_examples=50)
def test_api_target_dummy_fi(case, app):
    case.path_parameters = {"project_id": "dummy-fi"}
    response = case.call_wsgi(app)
    case.validate_response(response)
