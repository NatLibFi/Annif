"""Vocabulary management functionality for Annif"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING

import annif
import annif.corpus
import annif.util
from annif.datadir import DatadirMixin
from annif.exception import NotInitializedException

from .skos import VocabFileSKOS
from .subject_index import SubjectIndexFile
from .types import Subject, SubjectIndex, VocabSource

if TYPE_CHECKING:
    from rdflib.graph import Graph


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

    def _create_subject_index(self, vocab_source: VocabSource) -> SubjectIndex:
        subjects = SubjectIndexFile()
        subjects.load_subjects(vocab_source)
        annif.util.atomic_save(subjects, self.datadir, self.INDEX_FILENAME_CSV)
        return subjects

    def _update_subject_index(self, vocab_source: VocabSource) -> SubjectIndex:
        old_subjects = self.subjects
        new_subjects = SubjectIndexFile()
        new_subjects.load_subjects(vocab_source)
        updated_subjects = SubjectIndexFile()

        for old_subject in old_subjects:
            if new_subjects.contains_uri(old_subject.uri):
                new_subject = new_subjects[new_subjects.by_uri(old_subject.uri)]
            else:  # subject removed from new corpus
                new_subject = Subject(uri=old_subject.uri, labels=None, notation=None)
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
    def skos(self) -> VocabFileSKOS:
        """return the subject vocabulary from SKOS file"""
        if self._skos_vocab is not None:
            return self._skos_vocab

        # attempt to load graph from dump file
        dumppath = os.path.join(self.datadir, self.INDEX_FILENAME_DUMP)
        if os.path.exists(dumppath):
            logger.debug(f"loading graph dump from {dumppath}")
            try:
                self._skos_vocab = VocabFileSKOS(dumppath)
            except ModuleNotFoundError:
                # Probably dump has been saved using a different rdflib version
                logger.debug("could not load graph dump, using turtle file")
            else:
                return self._skos_vocab

        # graph dump file not found - parse ttl file instead
        path = os.path.join(self.datadir, self.INDEX_FILENAME_TTL)
        if os.path.exists(path):
            logger.debug(f"loading graph from {path}")
            self._skos_vocab = VocabFileSKOS(path)
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
        vocab_source: VocabSource,
        force: bool = False,
    ) -> None:
        """Load subjects from a subject corpus and save them into one
        or more subject index files as well as a SKOS/Turtle file for later
        use. If force=True, replace the existing subject index completely."""

        if not force and os.path.exists(
            os.path.join(self.datadir, self.INDEX_FILENAME_CSV)
        ):
            logger.info("updating existing subject index")
            self._subjects = self._update_subject_index(vocab_source)
        else:
            logger.info("creating subject index")
            self._subjects = self._create_subject_index(vocab_source)

        skosfile = os.path.join(self.datadir, self.INDEX_FILENAME_TTL)
        logger.info(f"saving vocabulary into SKOS file {skosfile}")
        vocab_source.save_skos(skosfile)

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
