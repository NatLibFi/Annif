"""Support for subjects loaded from a SKOS/RDF file"""

import os.path
import shutil
import joblib
import rdflib
import rdflib.util
from rdflib.namespace import SKOS, RDF, OWL
import annif.util
from .types import Subject, SubjectCorpus


def serialize_subjects_to_skos(subjects, language, path):
    """Create a SKOS representation of the given subjects and serialize it
    into a SKOS/Turtle file with the given path name."""

    graph = rdflib.Graph()
    graph.namespace_manager.bind('skos', SKOS)
    for subject in subjects:
        graph.add((rdflib.URIRef(subject.uri), RDF.type, SKOS.Concept))
        graph.add((rdflib.URIRef(subject.uri),
                   SKOS.prefLabel,
                   rdflib.Literal(subject.label, language)))
        graph.add((rdflib.URIRef(subject.uri),
                   SKOS.notation,
                   rdflib.Literal(subject.notation)))
    graph.serialize(destination=path, format='turtle')
    # also dump the graph in joblib format which is faster to load
    annif.util.atomic_save(graph,
                           *os.path.split(path.replace('.ttl', '.dump.gz')),
                           method=joblib.dump)


class SubjectFileSKOS(SubjectCorpus):
    """A subject corpus that uses SKOS files"""

    def __init__(self, path, language):
        self.path = path
        self.language = language
        if path.endswith('.dump.gz'):
            self.graph = joblib.load(path)
        else:
            self.graph = rdflib.Graph()
            self.graph.load(self.path,
                            format=rdflib.util.guess_format(self.path))

    @property
    def subjects(self):
        for concept in self.concepts:
            labels = self.graph.preferredLabel(concept, lang=self.language)
            notation = self.graph.value(concept, SKOS.notation, None, any=True)
            if not labels:
                continue
            label = str(labels[0][1])
            if notation is not None:
                notation = str(notation)
            yield Subject(uri=str(concept), label=label, notation=notation,
                          text=None)

    @property
    def concepts(self):
        for concept in self.graph.subjects(RDF.type, SKOS.Concept):
            if (concept, OWL.deprecated, rdflib.Literal(True)) in self.graph:
                continue
            yield concept

    def get_concept_labels(self, concept, label_types, language):
        return [str(label)
                for label_type in label_types
                for label in self.graph.objects(concept, label_type)
                if label.language == language]

    @staticmethod
    def is_rdf_file(path):
        """return True if the path looks like an RDF file that can be loaded
        as SKOS"""

        fmt = rdflib.util.guess_format(path)
        return fmt is not None

    def save_skos(self, path, language):
        """Save the contents of the subject vocabulary into a SKOS/Turtle
        file with the given path name."""

        if self.path.endswith('.ttl'):
            # input is already in Turtle syntax, no need to reserialize
            if not os.path.exists(path) or \
               not os.path.samefile(self.path, path):
                shutil.copyfile(self.path, path)
        else:
            # need to serialize into Turtle
            self.graph.serialize(destination=path, format='turtle')
        # also dump the graph in joblib format which is faster to load
        annif.util.atomic_save(self.graph,
                               *os.path.split(
                                   path.replace('.ttl', '.dump.gz')),
                               method=joblib.dump)
