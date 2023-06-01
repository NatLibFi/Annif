"""Support for subjects loaded from a SKOS/RDF file"""
from __future__ import annotations

import collections
import os.path
import shutil
from typing import TYPE_CHECKING

import rdflib
import rdflib.util
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import annif.util

from .types import Subject, SubjectCorpus

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rdflib.term import URIRef


def serialize_subjects_to_skos(subjects: Iterator, path: str) -> None:
    """Create a SKOS representation of the given subjects and serialize it
    into a SKOS/Turtle file with the given path name."""
    import joblib

    graph = rdflib.Graph()
    graph.namespace_manager.bind("skos", SKOS)
    for subject in subjects:
        graph.add((rdflib.URIRef(subject.uri), RDF.type, SKOS.Concept))
        for lang, label in subject.labels.items():
            graph.add(
                (
                    rdflib.URIRef(subject.uri),
                    SKOS.prefLabel,
                    rdflib.Literal(label, lang),
                )
            )
        graph.add(
            (
                rdflib.URIRef(subject.uri),
                SKOS.notation,
                rdflib.Literal(subject.notation),
            )
        )
    graph.serialize(destination=path, format="turtle")
    # also dump the graph in joblib format which is faster to load
    annif.util.atomic_save(
        graph, *os.path.split(path.replace(".ttl", ".dump.gz")), method=joblib.dump
    )


class SubjectFileSKOS(SubjectCorpus):
    """A subject corpus that uses SKOS files"""

    PREF_LABEL_PROPERTIES = (SKOS.prefLabel, RDFS.label)

    _languages = None

    def __init__(self, path: str) -> None:
        self.path = path
        if path.endswith(".dump.gz"):
            import joblib

            self.graph = joblib.load(path)
        else:
            self.graph = rdflib.Graph()
            self.graph.parse(self.path, format=rdflib.util.guess_format(self.path))

    @property
    def languages(self) -> set[str]:
        if self._languages is None:
            self._languages = {
                label.language
                for concept in self.concepts
                for label_type in self.PREF_LABEL_PROPERTIES
                for label in self.graph.objects(concept, label_type)
                if label.language is not None
            }
        return self._languages

    def _concept_labels(self, concept: URIRef) -> dict[str, str]:
        by_lang = self.get_concept_labels(concept, self.PREF_LABEL_PROPERTIES)
        return {
            lang: by_lang[lang][0]
            if by_lang[lang]  # correct lang
            else by_lang[None][0]
            if by_lang[None]  # no language
            else self.graph.namespace_manager.qname(concept)
            for lang in self.languages
        }

    @property
    def subjects(self) -> Iterator[Subject]:
        for concept in self.concepts:
            labels = self._concept_labels(concept)

            notation = self.graph.value(concept, SKOS.notation, None, any=True)
            if notation is not None:
                notation = str(notation)

            yield Subject(uri=str(concept), labels=labels, notation=notation)

    @property
    def concepts(self) -> Iterator[URIRef]:
        for concept in self.graph.subjects(RDF.type, SKOS.Concept):
            if (concept, OWL.deprecated, rdflib.Literal(True)) in self.graph:
                continue
            yield concept

    def get_concept_labels(
        self,
        concept: URIRef,
        label_types: Sequence[URIRef],
    ) -> collections.defaultdict[str | None, list[str]]:
        """return all the labels of the given concept with the given label
        properties as a dict-like object where the keys are language codes
        and the values are lists of labels in that language"""
        labels_by_lang = collections.defaultdict(list)

        for label_type in label_types:
            for label in self.graph.objects(concept, label_type):
                labels_by_lang[label.language].append(str(label))

        return labels_by_lang

    @staticmethod
    def is_rdf_file(path: str) -> bool:
        """return True if the path looks like an RDF file that can be loaded
        as SKOS"""

        fmt = rdflib.util.guess_format(path)
        return fmt is not None

    def save_skos(self, path: str) -> None:
        """Save the contents of the subject vocabulary into a SKOS/Turtle
        file with the given path name."""

        if self.path.endswith(".ttl"):
            # input is already in Turtle syntax, no need to reserialize
            if not os.path.exists(path) or not os.path.samefile(self.path, path):
                shutil.copyfile(self.path, path)
        else:
            # need to serialize into Turtle
            self.graph.serialize(destination=path, format="turtle")
        # also dump the graph in joblib format which is faster to load
        import joblib

        annif.util.atomic_save(
            self.graph,
            *os.path.split(path.replace(".ttl", ".dump.gz")),
            method=joblib.dump,
        )
