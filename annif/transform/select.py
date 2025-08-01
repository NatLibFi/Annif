"""A transformation that allows selecting Document metadata fields to be
used instead of the main text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annif.corpus import Document

from . import transform

if TYPE_CHECKING:
    from annif.project import AnnifProject


class SelectTransform(transform.BaseTransform):
    name = "select"

    def __init__(self, project: AnnifProject | None, *fields: str) -> None:
        super().__init__(project)
        self.fields = fields

    def _get_texts(self, doc):
        for fld in self.fields:
            if fld == "text":
                yield doc.text
            else:
                yield doc.metadata.get(fld, "")

    def transform_doc(self, doc: Document) -> Document:
        new_text = "\n".join(self._get_texts(doc))
        return Document(
            text=new_text, subject_set=doc.subject_set, metadata=doc.metadata
        )
