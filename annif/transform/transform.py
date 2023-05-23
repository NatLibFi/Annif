"""Common functionality for transforming text of input documents."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

from annif.corpus import TransformingDocumentCorpus
from annif.exception import ConfigurationException

if TYPE_CHECKING:
    from unittest.mock import Mock

    from annif.corpus import DocumentCorpus
    from annif.project import AnnifProject
    from annif.transform.inputlimiter import InputLimiter
    from annif.transform.langfilter import LangFilter


class BaseTransform(metaclass=abc.ABCMeta):
    """Base class for text transformations, which need to implement the
    transform function."""

    name = None

    def __init__(self, project: Optional[Union[AnnifProject, Mock]]) -> None:
        self.project = project

    @abc.abstractmethod
    def transform_fn(self, text):
        """Perform the text transformation."""
        pass  # pragma: no cover


class IdentityTransform(BaseTransform):
    """Transform that does not modify text but simply passes it through."""

    name = "pass"

    def transform_fn(self, text: str) -> str:
        return text


class TransformChain:
    """Class instantiating and holding the transformation objects performing
    the actual text transformation."""

    def __init__(
        self,
        transform_classes: List[
            Union[Type[InputLimiter], Type[IdentityTransform], Type[LangFilter]]
        ],
        args: List[
            Union[
                Tuple[List[Any], Dict[str, str]],
                Tuple[List[str], Dict[Any, Any]],
                Tuple[List[Any], Dict[Any, Any]],
            ]
        ],
        project: Optional[Union[AnnifProject, Mock]],
    ) -> None:
        self.project = project
        self.transforms = self._init_transforms(transform_classes, args)

    def _init_transforms(
        self,
        transform_classes: List[
            Union[Type[InputLimiter], Type[IdentityTransform], Type[LangFilter]]
        ],
        args: List[
            Union[
                Tuple[List[Any], Dict[str, str]],
                Tuple[List[str], Dict[Any, Any]],
                Tuple[List[Any], Dict[Any, Any]],
            ]
        ],
    ) -> List[Union[InputLimiter, IdentityTransform, LangFilter]]:
        transforms = []
        for trans, (posargs, kwargs) in zip(transform_classes, args):
            try:
                transforms.append(trans(self.project, *posargs, **kwargs))
            except (ValueError, TypeError):
                raise ConfigurationException(
                    f"Invalid arguments to {trans.name} transform: "
                    f"{posargs}, {kwargs})",
                    project_id=self.project.project_id,
                )
        return transforms

    def transform_text(self, text: str) -> str:
        for trans in self.transforms:
            text = trans.transform_fn(text)
        return text

    def transform_corpus(self, corpus: DocumentCorpus) -> TransformingDocumentCorpus:
        return TransformingDocumentCorpus(corpus, self.transform_text)
