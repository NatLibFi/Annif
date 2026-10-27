"""Unit tests for the custom OpenAPI form data request body validators"""

import asyncio

import pytest
from connexion.exceptions import ClientProblem
from starlette.formparsers import MultiPartException
from starlette.requests import Request

from annif.openapi.validation import CustomMultiPartFormDataValidator


def _multipart_validator(max_part_size: int | None):
    """a multipart form data validator with the given max_part_size set; use
    a local subclass so that the limit is not changed on the shared class"""
    limit = max_part_size

    class LimitedMultiPartFormDataValidator(CustomMultiPartFormDataValidator):
        max_part_size = limit

    # only the arguments needed by _parse() are given; the tests do not use
    # the schema validation
    return LimitedMultiPartFormDataValidator(
        schema={}, encoding="utf-8", strict_validation=False
    )


def _multipart_request(value: str, *, malformed: bool = False) -> Request:
    """a Starlette request with a multipart body containing the given value"""
    boundary = "annif-test-boundary"
    content_type = (
        b"multipart/form-data"
        if malformed
        else f"multipart/form-data; boundary={boundary}".encode()
    )
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="text"\r\n'
        f"\r\n"
        f"{value}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    scope = {
        "type": "http",
        "headers": [
            (b"content-type", content_type),
            (b"content-length", str(len(body)).encode()),
        ],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def test_multipart_validator_rejects_oversized_field_with_413():
    validator = _multipart_validator(max_part_size=100)
    request = _multipart_request("A" * 1000)
    with pytest.raises(ClientProblem) as exc_info:
        asyncio.run(validator._parse(request.stream(), request.scope))
    assert exc_info.value.status_code == 413


def test_multipart_validator_reraises_non_size_errors():
    """MultiPartExceptions that are not about the maximum part size (e.g. a
    missing boundary) must not be masked by the 413 handler."""
    validator = _multipart_validator(max_part_size=None)
    request = _multipart_request("example", malformed=True)
    with pytest.raises(MultiPartException):
        asyncio.run(validator._parse(request.stream(), request.scope))
