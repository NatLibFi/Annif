"""Custom validators for the Annif API."""

from __future__ import annotations

import logging
import typing as t
from typing import Any

from connexion.exceptions import BadRequestProblem, ClientProblem
from connexion.json_schema import format_error_with_path
from connexion.validators import (
    FormDataValidator,
    JSONRequestBodyValidator,
    MultiPartFormDataValidator,
)
from jsonschema.exceptions import ValidationError
from starlette.formparsers import MultiPartException
from starlette.types import Scope

logger = logging.getLogger("openapi.validation")


class CustomRequestBodyValidator(JSONRequestBodyValidator):
    """Custom request body validator that overrides the default error message for the
    'maxItems' validator for the 'documents' property to prevent logging request body
    with the contents of all documents."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _validate(self, body: Any) -> dict | None:
        try:
            return self._validator.validate(body)
        except ValidationError as exception:
            if exception.validator == "maxItems" and list(exception.schema_path) == [
                "properties",
                "documents",
                "maxItems",
            ]:
                exception.message = "too many items"
            error_path_msg = format_error_with_path(exception=exception)
            logger.error(
                f"Validation error: {exception.message}{error_path_msg}",
                extra={"validator": "body"},
            )
            raise BadRequestProblem(detail=f"{exception.message}{error_path_msg}")


class CustomFormDataValidator(FormDataValidator):
    """Form data validator for application/x-www-form-urlencoded requests
    that uses a configurable maximum part size. Starlette's form parsers
    default to a hard-coded 1 MB limit per form field, which is lower than
    Annif's overall content limit and would reject otherwise valid requests
    with a 500 error. The limit is set on the class-level ``max_part_size``
    attribute when the Connexion app is created, and exceeding it results in
    a 413 Payload Too Large response instead of an unhandled
    MultiPartException."""

    #: maximum size (in bytes) of a single form field, set by
    #: ``annif.create_cx_app()`` from the app configuration; ``None`` means
    #: fall back to Starlette's default limit
    max_part_size: int | None = None

    @property
    def _form_parser_cls(self) -> type:
        # resolve the parser class from the concrete subclass (FormParser for
        # application/x-www-form-urlencoded, MultiPartParser for
        # multipart/form-data) and wrap it to pass the maximum part size
        base_cls = super()._form_parser_cls
        max_part_size = type(self).max_part_size
        if max_part_size is None:
            return base_cls

        class LimitedParser(base_cls):  # type: ignore[valid-type,misc]
            def __init__(self, headers, stream, **parser_kwargs):
                parser_kwargs["max_part_size"] = max_part_size
                super().__init__(headers, stream, **parser_kwargs)

        return LimitedParser

    async def _parse(self, stream: t.AsyncGenerator[bytes, None], scope: Scope) -> dict:
        try:
            return await super()._parse(stream, scope)
        except MultiPartException as err:
            raise ClientProblem(
                status=413,
                title="Payload Too Large",
                detail=f"Form field exceeded maximum size: {err.message}",
            ) from err


class CustomMultiPartFormDataValidator(
    CustomFormDataValidator, MultiPartFormDataValidator
):
    """Form data validator for multipart/form-data requests with a
    configurable maximum part size. Inherits from MultiPartFormDataValidator
    so that the correct (multipart) parser class is selected."""
