"""Support for subjects loaded from a SKOS/RDF file"""

import rdflib
import rdflib.util
from rdflib.namespace import SKOS, RDF, OWL
from .subject import Subject, SubjectCorpus


class SubjectFileSKOS(SubjectCorpus):
    """A subject corpus that uses SKOS files"""

    def __init__(self, path, language):
        self.path = path
        self.language = language

    @property
    def subjects(self):
        graph = rdflib.Graph()
        graph.load(self.path, format=rdflib.util.guess_format(self.path))
        for concept in graph.subjects(RDF.type, SKOS.Concept):
            if (concept, OWL.deprecated, rdflib.Literal(True)) in graph:
                continue
            labels = graph.preferredLabel(concept, lang=self.language)
            if not labels:
                continue
            label = str(labels[0][1])
            yield Subject(uri=str(concept), label=label, text=None)

    @staticmethod
    def is_rdf_file(path):
        """return True if the path looks like an RDF file that can be loaded
        as SKOS"""

        fmt = rdflib.util.guess_format(path)
        return fmt is not None
