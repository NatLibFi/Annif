"""A simple transformation that truncates the text of input documents to a
given character length."""
from __future__ import annotations

from typing import TYPE_CHECKING

from annif.exception import ConfigurationException

from . import transform

if TYPE_CHECKING:
    from annif.project import AnnifProject


class InputLimiter(transform.BaseTransform):
    name = "limit"

    def __init__(self, project: AnnifProject | None, input_limit: str) -> None:
        super().__init__(project)
        self.input_limit = int(input_limit)
        self._validate_value(self.input_limit)

    def transform_fn(self, text: str) -> str:
        return text[: self.input_limit]

    def _validate_value(self, input_limit: int) -> None:
        if input_limit < 0:
            raise ConfigurationException(
                "input_limit in limit_input transform cannot be negative",
                project_id=self.project.project_id,
            )
