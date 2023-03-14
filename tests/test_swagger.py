"""Unit tests for Annif REST API / Swagger spec"""
import schemathesis

schema = schemathesis.from_path("annif/openapi/annif.yaml")


@schema.parametrize()
def test_api(case, app):
    response = case.call_wsgi(app)
    case.validate_response(response)
