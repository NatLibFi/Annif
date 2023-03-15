"""Unit tests for Annif REST API / Swagger spec"""
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
