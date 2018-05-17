"""Support for subjects loaded from a SKOS/RDF file"""

import rdflib
import rdflib.util
from rdflib.namespace import SKOS, RDF, OWL
from .subject import SubjectIndex, Subject


def skos_file_as_corpus(path, language):
    graph = rdflib.Graph()
    graph.load(path, format=rdflib.util.guess_format(path))
    for concept in graph.subjects(RDF.type, SKOS.Concept):
        if (concept, OWL.deprecated, rdflib.Literal(True)) in graph:
            continue
        labels = graph.preferredLabel(concept, lang=language)
        if len(labels) == 0:
            continue
        label = str(labels[0][1])
        yield Subject(uri=str(concept), label=label, text=None)


class SubjectIndexSKOS (SubjectIndex):
    """A subject index that uses SKOS files instead of TSV files"""

    @classmethod
    def is_rdf_file(cls, path):
        """return True if the path looks like an RDF file that can be loaded
        as SKOS"""

        format = rdflib.util.guess_format(path)
        return format is not None

    @classmethod
    def load(cls, path, language):
        """Load subjects from a SKOS file and return a subject index."""

        return cls(skos_file_as_corpus(path, language))
