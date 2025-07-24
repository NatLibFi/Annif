"""Common functionality for transforming text of input documents."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Type

from annif.corpus import Document, TransformingDocumentCorpus
from annif.exception import ConfigurationException

if TYPE_CHECKING:
    from annif.corpus.types import DocumentCorpus
    from annif.project import AnnifProject


class BaseTransform(abc.ABC):
    """Base class for text transformations, which need to implement either the
    transform_doc function or the transform_text function."""

    name = None

    def __init__(self, project: AnnifProject | None) -> None:
        self.project = project
        if (
            type(self).transform_text == BaseTransform.transform_text
            and type(self).transform_doc == BaseTransform.transform_doc
        ):
            raise NotImplementedError(
                "Subclasses must override transform_text or transform_doc"
            )

    def transform_doc(self, doc: Document) -> Document:
        """Perform a transformation on a Document. By default, only the text is
        transformed by calling self.transform_text()."""

        transformed_text = self.transform_text(doc.text)
        return Document(
            text=transformed_text, subject_set=doc.subject_set, metadata=doc.metadata
        )

    def transform_text(self, text: str) -> str:
        """Perform a transformation on the document text."""

        raise NotImplementedError(
            "Subclasses must implement transform_text if they call it"
        )


class IdentityTransform(BaseTransform):
    """Transform that does not modify the document but simply passes it through."""

    name = "pass"

    def transform_text(self, text: str) -> str:
        return text


class TransformChain:
    """Class instantiating and holding the transformation objects performing
    the actual text transformation."""

    def __init__(
        self,
        transform_classes: list[Type[BaseTransform]],
        args: list[tuple[list, dict]],
        project: AnnifProject | None,
    ) -> None:
        self.project = project
        self.transforms = self._init_transforms(transform_classes, args)

    def _init_transforms(
        self,
        transform_classes: list[Type[BaseTransform]],
        args: list[tuple[list, dict]],
    ) -> list[BaseTransform]:
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

    def transform_doc(self, doc: Document) -> Document:
        for trans in self.transforms:
            doc = trans.transform_doc(doc)
        return doc

    def transform_corpus(self, corpus: DocumentCorpus) -> TransformingDocumentCorpus:
        return TransformingDocumentCorpus(corpus, self.transform_doc)
