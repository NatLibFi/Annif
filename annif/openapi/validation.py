"""Custom validator for the Annif API."""
from __future__ import annotations

import logging

import jsonschema
from connexion import decorators
from connexion.exceptions import BadRequestProblem
from connexion.utils import is_null

logger = logging.getLogger("openapi.validation")


class CustomRequestBodyValidator(decorators.validation.RequestBodyValidator):
    """Custom request body validator that overrides the default error message for the
    'maxItems' validator for the 'documents' property."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def validate_schema(
        self,
        data: list | dict,
        url: str,
    ) -> None:
        """Validate the request body against the schema."""

        if self.is_null_value_valid and is_null(data):
            return None  # pragma: no cover

        try:
            self.validator.validate(data)
        except jsonschema.ValidationError as exception:
            if exception.validator == "maxItems" and list(exception.schema_path) == [
                "properties",
                "documents",
                "maxItems",
            ]:
                exception.message = "too many items"

            error_path_msg = self._error_path_message(exception=exception)
            logger.error(
                "{url} validation error: {error}{error_path_msg}".format(
                    url=url, error=exception.message, error_path_msg=error_path_msg
                ),
                extra={"validator": "body"},
            )
            raise BadRequestProblem(detail=f"{exception.message}{error_path_msg}")
        return None
