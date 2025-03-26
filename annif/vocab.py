"""Vocabulary management functionality for Annif"""

from __future__ import annotations

import abc
import csv
import os.path
from typing import TYPE_CHECKING

import annif
import annif.corpus
import annif.util
from annif.datadir import DatadirMixin
from annif.exception import NotInitializedException

if TYPE_CHECKING:
    from rdflib.graph import Graph

    from annif.corpus.skos import SubjectFileSKOS
    from annif.corpus.subject import Subject, SubjectCorpus


logger = annif.logger
logger.addFilter(annif.util.DuplicateFilter())


class AnnifVocabulary(DatadirMixin):
    """Class representing a subject vocabulary which can be used by multiple
    Annif projects."""

    # defaults for uninitialized instances
    _subjects = None

    # constants
    INDEX_FILENAME_DUMP = "subjects.dump.gz"
    INDEX_FILENAME_TTL = "subjects.ttl"
    INDEX_FILENAME_CSV = "subjects.csv"

    def __init__(self, vocab_id: str, datadir: str) -> None:
        DatadirMixin.__init__(self, datadir, "vocabs", vocab_id)
        self.vocab_id = vocab_id
        self._skos_vocab = None

    def _create_subject_index(self, subject_corpus: SubjectCorpus) -> SubjectIndex:
        subjects = SubjectIndexFile()
        subjects.load_subjects(subject_corpus)
        annif.util.atomic_save(subjects, self.datadir, self.INDEX_FILENAME_CSV)
        return subjects

    def _update_subject_index(self, subject_corpus: SubjectCorpus) -> SubjectIndex:
        old_subjects = self.subjects
        new_subjects = SubjectIndexFile()
        new_subjects.load_subjects(subject_corpus)
        updated_subjects = SubjectIndexFile()

        for old_subject in old_subjects:
            if new_subjects.contains_uri(old_subject.uri):
                new_subject = new_subjects[new_subjects.by_uri(old_subject.uri)]
            else:  # subject removed from new corpus
                new_subject = annif.corpus.Subject(
                    uri=old_subject.uri, labels=None, notation=None
                )
            updated_subjects.append(new_subject)
        for new_subject in new_subjects:
            if not old_subjects.contains_uri(new_subject.uri):
                updated_subjects.append(new_subject)
        annif.util.atomic_save(updated_subjects, self.datadir, self.INDEX_FILENAME_CSV)
        return updated_subjects

    @property
    def subjects(self) -> SubjectIndex:
        if self._subjects is None:
            path = os.path.join(self.datadir, self.INDEX_FILENAME_CSV)
            if os.path.exists(path):
                logger.debug("loading subjects from %s", path)
                self._subjects = SubjectIndexFile.load(path)
            else:
                raise NotInitializedException("subject file {} not found".format(path))
        return self._subjects

    @property
    def skos(self) -> SubjectFileSKOS:
        """return the subject vocabulary from SKOS file"""
        if self._skos_vocab is not None:
            return self._skos_vocab

        # attempt to load graph from dump file
        dumppath = os.path.join(self.datadir, self.INDEX_FILENAME_DUMP)
        if os.path.exists(dumppath):
            logger.debug(f"loading graph dump from {dumppath}")
            try:
                self._skos_vocab = annif.corpus.SubjectFileSKOS(dumppath)
            except ModuleNotFoundError:
                # Probably dump has been saved using a different rdflib version
                logger.debug("could not load graph dump, using turtle file")
            else:
                return self._skos_vocab

        # graph dump file not found - parse ttl file instead
        path = os.path.join(self.datadir, self.INDEX_FILENAME_TTL)
        if os.path.exists(path):
            logger.debug(f"loading graph from {path}")
            self._skos_vocab = annif.corpus.SubjectFileSKOS(path)
            # store the dump file so we can use it next time
            self._skos_vocab.save_skos(path)
            return self._skos_vocab

        raise NotInitializedException(f"graph file {path} not found")

    def __len__(self) -> int:
        return len(self.subjects)

    @property
    def languages(self) -> list[str]:
        try:
            return self.subjects.languages
        except NotInitializedException:
            return []

    def load_vocabulary(
        self,
        subject_corpus: SubjectCorpus,
        force: bool = False,
    ) -> None:
        """Load subjects from a subject corpus and save them into one
        or more subject index files as well as a SKOS/Turtle file for later
        use. If force=True, replace the existing subject index completely."""

        if not force and os.path.exists(
            os.path.join(self.datadir, self.INDEX_FILENAME_CSV)
        ):
            logger.info("updating existing subject index")
            self._subjects = self._update_subject_index(subject_corpus)
        else:
            logger.info("creating subject index")
            self._subjects = self._create_subject_index(subject_corpus)

        skosfile = os.path.join(self.datadir, self.INDEX_FILENAME_TTL)
        logger.info(f"saving vocabulary into SKOS file {skosfile}")
        subject_corpus.save_skos(skosfile)

    def as_graph(self) -> Graph:
        """return the vocabulary as an rdflib graph"""
        return self.skos.graph

    def dump(self) -> dict[str, str | list | int | bool]:
        """return this vocabulary as a dict"""

        try:
            languages = list(sorted(self.languages))
            size = len(self)
            loaded = True
        except NotInitializedException:
            languages = []
            size = None
            loaded = False

        return {
            "vocab_id": self.vocab_id,
            "languages": languages,
            "size": size,
            "loaded": loaded,
        }


class SubjectIndex(metaclass=abc.ABCMeta):
    """Base class for an index that remembers the associations between
    integer subject IDs and their URIs and labels."""

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def languages(self) -> list[str] | None:
        pass

    @abc.abstractmethod
    def __getitem__(self, subject_id: int) -> Subject:
        pass

    @abc.abstractmethod
    def contains_uri(self, uri: str) -> bool:
        pass

    @abc.abstractmethod
    def by_uri(self, uri: str, warnings: bool = True) -> int | None:
        """return the subject ID of a subject by its URI, or None if not found.
        If warnings=True, log a warning message if the URI cannot be found."""
        pass

    @abc.abstractmethod
    def by_label(self, label: str | None, language: str) -> int | None:
        """return the subject ID of a subject by its label in a given
        language"""
        pass

    @abc.abstractmethod
    def active(self) -> list[tuple[int, Subject]]:
        """return a list of (subject_id, subject) tuples of all subjects that
        are not deprecated"""
        pass


class SubjectIndexFile(SubjectIndex):
    def __init__(self) -> None:
        self._subjects = []
        self._uri_idx = {}
        self._label_idx = {}
        self._languages = None

    def load_subjects(self, corpus: SubjectCorpus) -> None:
        """Initialize the subject index from a subject corpus"""

        self._languages = corpus.languages
        for subject in corpus.subjects:
            self.append(subject)

    def __len__(self) -> int:
        return len(self._subjects)

    @property
    def languages(self) -> list[str] | None:
        return self._languages

    def __getitem__(self, subject_id: int) -> Subject:
        return self._subjects[subject_id]

    def append(self, subject: Subject) -> None:
        if self._languages is None and subject.labels is not None:
            self._languages = list(subject.labels.keys())

        subject_id = len(self._subjects)
        self._uri_idx[subject.uri] = subject_id
        if subject.labels:
            for lang, label in subject.labels.items():
                self._label_idx[(label, lang)] = subject_id
        self._subjects.append(subject)

    def contains_uri(self, uri: str) -> bool:
        return uri in self._uri_idx

    def by_uri(self, uri: str, warnings: bool = True) -> int | None:
        try:
            return self._uri_idx[uri]
        except KeyError:
            if warnings:
                logger.warning("Unknown subject URI <%s>", uri)
            return None

    def by_label(self, label: str | None, language: str) -> int | None:
        try:
            return self._label_idx[(label, language)]
        except KeyError:
            logger.warning('Unknown subject label "%s"@%s', label, language)
            return None

    def deprecated_ids(self) -> list[int]:
        """return indices of deprecated subjects"""

        return [
            subject_id
            for subject_id, subject in enumerate(self._subjects)
            if subject.labels is None
        ]

    @property
    def active(self) -> list[tuple[int, Subject]]:
        return [
            (subj_id, subject)
            for subj_id, subject in enumerate(self._subjects)
            if subject.labels is not None
        ]

    def save(self, path: str) -> None:
        """Save this subject index into a file with the given path name."""

        fieldnames = ["uri", "notation"] + [f"label_{lang}" for lang in self._languages]

        with open(path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for subject in self:
                row = {"uri": subject.uri, "notation": subject.notation or ""}
                if subject.labels:
                    for lang, label in subject.labels.items():
                        row[f"label_{lang}"] = label
                writer.writerow(row)

    @classmethod
    def load(cls, path: str) -> SubjectIndex:
        """Load a subject index from a CSV file and return it."""

        corpus = annif.corpus.SubjectFileCSV(path)
        subject_index = cls()
        subject_index.load_subjects(corpus)
        return subject_index
