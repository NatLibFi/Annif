"""Vocabulary management functionality for Annif"""

import os.path
import re
import annif
import annif.corpus
import annif.util
from annif.datadir import DatadirMixin
from annif.exception import NotInitializedException
from annif.util import parse_args

logger = annif.logger


def get_vocab(vocab_spec, datadir, default_language):
    match = re.match(r'(\w+)(\((.*)\))?', vocab_spec)
    if match is None:
        raise ValueError(f"Invalid vocabulary specification: {vocab_spec}")
    vocab_id = match.group(1)
    posargs, kwargs = parse_args(match.group(3))
    language = posargs[0] if posargs else default_language

    return AnnifVocabulary(vocab_id, datadir, language)


class AnnifVocabulary(DatadirMixin):
    """Class representing a subject vocabulary which can be used by multiple
    Annif projects."""

    # defaults for uninitialized instances
    _subjects = None

    # constants
    INDEX_FILENAME_DUMP = "subjects.dump.gz"
    INDEX_FILENAME_TTL = "subjects.ttl"
    INDEX_FILENAME_CSV = "subjects.csv"

    def __init__(self, vocab_id, datadir, language):
        DatadirMixin.__init__(self, datadir, 'vocabs', vocab_id)
        self.vocab_id = vocab_id
        self.language = language
        self._skos_vocab = None

    def _create_subject_index(self, subject_corpus):
        subjects = annif.corpus.SubjectIndex()
        subjects.load_subjects(subject_corpus)
        annif.util.atomic_save(subjects, self.datadir,
                               self.INDEX_FILENAME_CSV)
        return subjects

    def _update_subject_index(self, subject_corpus):
        old_subjects = self.subjects
        new_subjects = annif.corpus.SubjectIndex()
        new_subjects.load_subjects(subject_corpus)
        updated_subjects = annif.corpus.SubjectIndex()

        for old_subject in old_subjects:
            if new_subjects.contains_uri(old_subject.uri):
                new_subject = new_subjects[new_subjects.by_uri(
                    old_subject.uri)]
            else:  # subject removed from new corpus
                new_subject = annif.corpus.Subject(uri=old_subject.uri,
                                                   labels=None,
                                                   notation=None)
            updated_subjects.append(new_subject)
        for new_subject in new_subjects:
            if not old_subjects.contains_uri(new_subject.uri):
                updated_subjects.append(new_subject)
        annif.util.atomic_save(updated_subjects, self.datadir,
                               self.INDEX_FILENAME_CSV)
        return updated_subjects

    @property
    def subjects(self):
        if self._subjects is None:
            path = os.path.join(self.datadir, self.INDEX_FILENAME_CSV)
            if os.path.exists(path):
                logger.debug('loading subjects from %s', path)
                self._subjects = annif.corpus.SubjectIndex.load(path)
            else:
                raise NotInitializedException(
                    "subject file {} not found".format(path))
        return self._subjects

    @property
    def skos(self):
        """return the subject vocabulary from SKOS file"""
        if self._skos_vocab is not None:
            return self._skos_vocab

        # attempt to load graph from dump file
        dumppath = os.path.join(self.datadir, self.INDEX_FILENAME_DUMP)
        if os.path.exists(dumppath):
            logger.debug(f'loading graph dump from {dumppath}')
            try:
                self._skos_vocab = annif.corpus.SubjectFileSKOS(dumppath)
            except ModuleNotFoundError:
                # Probably dump has been saved using a different rdflib version
                logger.debug('could not load graph dump, using turtle file')
            else:
                return self._skos_vocab

        # graph dump file not found - parse ttl file instead
        path = os.path.join(self.datadir, self.INDEX_FILENAME_TTL)
        if os.path.exists(path):
            logger.debug(f'loading graph from {path}')
            self._skos_vocab = annif.corpus.SubjectFileSKOS(path)
            # store the dump file so we can use it next time
            self._skos_vocab.save_skos(path)
            return self._skos_vocab

        raise NotInitializedException(f'graph file {path} not found')

    def load_vocabulary(self, subject_corpus, force=False):
        """Load subjects from a subject corpus and save them into one
        or more subject index files as well as a SKOS/Turtle file for later
        use. If force=True, replace the existing subject index completely."""

        if not force and os.path.exists(
                os.path.join(self.datadir, self.INDEX_FILENAME_CSV)):
            logger.info('updating existing vocabulary')
            self._subjects = self._update_subject_index(subject_corpus)
        else:
            self._subjects = self._create_subject_index(subject_corpus)

        subject_corpus.save_skos(
            os.path.join(self.datadir, self.INDEX_FILENAME_TTL))

    def as_skos_file(self):
        """return the vocabulary as a file object, in SKOS/Turtle syntax"""
        return open(os.path.join(self.datadir, self.INDEX_FILENAME_TTL), 'rb')

    def as_graph(self):
        """return the vocabulary as an rdflib graph"""
        return self.skos.graph
