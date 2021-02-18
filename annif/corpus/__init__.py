"""Annif corpus operations"""


from .document import DocumentDirectory, DocumentFile, DocumentList, \
    TruncatingDocumentCorpus, LimitingDocumentCorpus
from .subject import Subject, SubjectFileTSV
from .subject import SubjectIndex, SubjectSet
from .skos import SubjectFileSKOS
from .types import Document
from .combine import CombinedCorpus

__all__ = ["DocumentDirectory", "DocumentFile", "DocumentList", "Subject",
           "SubjectFileTSV", "SubjectIndex", "SubjectSet", "SubjectFileSKOS",
           "Document", "CombinedCorpus", "TruncatingDocumentCorpus",
           "LimitingDocumentCorpus"]
