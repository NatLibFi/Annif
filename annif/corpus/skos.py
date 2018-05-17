"""Support for subjects loaded from a SKOS/RDF file"""

import rdflib
import rdflib.util
from rdflib.namespace import SKOS, RDF
from .subject import SubjectIndex, Subject


class SubjectIndexSKOS (SubjectIndex):
    """A subject index that uses SKOS files instead of TSV files"""

    @classmethod
    def load(cls, path, language):
        """Load subjects from a SKOS file and return a subject index."""

        def skos_file_as_corpus(path, language):
            graph = rdflib.Graph()
            format = rdflib.util.guess_format(path)
            graph.load(path, format=format)
            for concept in graph.subjects(RDF.type, SKOS.Concept):
                labels = graph.preferredLabel(concept, lang=language)
                if len(labels) > 0:
                    label = labels[0][1]
                    yield Subject(uri=str(concept), label=label, text=None)

        return cls(skos_file_as_corpus(path, language))
