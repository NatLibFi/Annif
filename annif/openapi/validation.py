"""Custom validator for the Annif API."""

from __future__ import annotations

import logging
from typing import Any

from connexion.exceptions import BadRequestProblem
from connexion.json_schema import format_error_with_path
from connexion.validators import JSONRequestBodyValidator
from jsonschema.exceptions import ValidationError

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
