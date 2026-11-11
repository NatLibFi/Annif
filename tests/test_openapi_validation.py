"""Unit tests for the custom OpenAPI form data request body validators"""

import asyncio
from urllib.parse import urlencode

import pytest
from connexion.exceptions import ClientProblem
from starlette.formparsers import MultiPartException
from starlette.requests import Request

from annif.openapi.validation import (
    CustomFormDataValidator,
    CustomMultiPartFormDataValidator,
)


def _multipart_body(value: str, *, malformed: bool = False) -> tuple[bytes, bytes]:
    """The body and Content-Type of a multipart request with the given value
    in the 'text' field"""
    body = (
        f"--annif-test-boundary\r\n"
        f'Content-Disposition: form-data; name="text"\r\n'
        f"\r\n"
        f"{value}\r\n"
        f"--annif-test-boundary--\r\n"
    ).encode()
    content_type = (
        b"multipart/form-data"
        if malformed
        else b"multipart/form-data; boundary=annif-test-boundary"
    )
    return body, content_type


def _form_body(fields: dict[str, str]) -> tuple[bytes, bytes]:
    """The body and Content-Type of an application/x-www-form-urlencoded
    request"""
    return urlencode(fields).encode(), b"application/x-www-form-urlencoded"


def _make_request(body: bytes, content_type: bytes) -> Request:
    """A Starlette request with the given body and Content-Type header"""
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


def _make_validator(validator_cls, max_part_size: int | None):
    """A validator of the given class with the given max_part_size set; a
    local subclass is used so that the limit is not changed on the shared
    class"""
    limit = max_part_size

    class LimitedValidator(validator_cls):
        max_part_size = limit

    # only the arguments needed by _parse() are given; the tests do not use
    # the schema validation
    return LimitedValidator(schema={}, encoding="utf-8", strict_validation=False)


def test_multipart_validator_rejects_oversized_field_with_413():
    validator = _make_validator(CustomMultiPartFormDataValidator, max_part_size=100)
    body, content_type = _multipart_body("A" * 1000)
    request = _make_request(body, content_type)
    with pytest.raises(ClientProblem) as exc_info:
        asyncio.run(validator._parse(request.stream(), request.scope))
    assert exc_info.value.status_code == 413


def test_form_validator_rejects_oversized_field_with_413():
    validator = _make_validator(CustomFormDataValidator, max_part_size=100)
    body, content_type = _form_body({"text": "A" * 1000})
    request = _make_request(body, content_type)
    with pytest.raises(ClientProblem) as exc_info:
        asyncio.run(validator._parse(request.stream(), request.scope))
    assert exc_info.value.status_code == 413


def test_multipart_validator_reraises_non_size_errors():
    validator = _make_validator(CustomMultiPartFormDataValidator, max_part_size=None)
    body, content_type = _multipart_body("example", malformed=True)
    request = _make_request(body, content_type)
    with pytest.raises(MultiPartException):
        asyncio.run(validator._parse(request.stream(), request.scope))


def test_form_validator_reraises_non_size_errors():
    validator = _make_validator(CustomFormDataValidator, max_part_size=None)
    body, content_type = _form_body({f"field{i:04d}": "value" for i in range(1001)})
    request = _make_request(body, content_type)
    with pytest.raises(MultiPartException):
        asyncio.run(validator._parse(request.stream(), request.scope))


def test_form_validator_parses_small_field():
    validator = _make_validator(CustomFormDataValidator, max_part_size=100)
    body, content_type = _form_body({"text": "example"})
    request = _make_request(body, content_type)
    data = asyncio.run(validator._parse(request.stream(), request.scope))
    assert data == {"text": ["example"]}
