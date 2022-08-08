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

    def __init__(self, vocab_id, datadir, language):
        DatadirMixin.__init__(self, datadir, 'vocabs', vocab_id)
        self.vocab_id = vocab_id
        self.language = language
        self._skos_vocab = None

    @staticmethod
    def _index_filename(language):
        return f"subjects.{language}.tsv"

    def _create_subject_index(self, subject_corpus, language):
        subjects = annif.corpus.SubjectIndex()
        subjects.load_subjects(subject_corpus, language)
        annif.util.atomic_save(subjects, self.datadir,
                               self._index_filename(language))
        return subjects

    def _update_subject_index(self, subject_corpus, language):
        old_subjects = self.subjects
        new_subjects = annif.corpus.SubjectIndex()
        new_subjects.load_subjects(subject_corpus, language)
        updated_subjects = annif.corpus.SubjectIndex()

        for uri, label, notation in old_subjects:
            if new_subjects.contains_uri(uri):
                label, notation = new_subjects[new_subjects.by_uri(uri)][1:3]
            else:  # subject removed from new corpus
                label, notation = None, None
            updated_subjects.append(uri, label, notation)
        for uri, label, notation in new_subjects:
            if not old_subjects.contains_uri(uri):
                updated_subjects.append(uri, label, notation)
        annif.util.atomic_save(updated_subjects, self.datadir,
                               self._index_filename(language))
        return updated_subjects

    @property
    def subjects(self):
        if self._subjects is None:
            path = os.path.join(self.datadir,
                                self._index_filename(self.language))
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
        dumppath = os.path.join(self.datadir, 'subjects.dump.gz')
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
        path = os.path.join(self.datadir, 'subjects.ttl')
        if os.path.exists(path):
            logger.debug(f'loading graph from {path}')
            self._skos_vocab = annif.corpus.SubjectFileSKOS(path)
            # store the dump file so we can use it next time
            self._skos_vocab.save_skos(path, self.language)
            return self._skos_vocab

        raise NotInitializedException(f'graph file {path} not found')

    def load_vocabulary(self, subject_corpus, default_language, force=False):
        """Load subjects from a subject corpus and save them into one
        or more subject index files as well as a SKOS/Turtle file for later
        use. If force=True, replace the existing subject index completely."""

        # default to language from project config if subject corpus isn't
        # language-aware or can't detect languages
        languages = subject_corpus.languages or [default_language]

        for language in languages:
            if not force and os.path.exists(
                    os.path.join(self.datadir,
                                 self._index_filename(language))):
                logger.info('updating existing vocabulary')
                subjects = self._update_subject_index(subject_corpus, language)
            else:
                subjects = self._create_subject_index(subject_corpus, language)

            if language == default_language:
                self._subjects = subjects

        subject_corpus.save_skos(os.path.join(self.datadir, 'subjects.ttl'),
                                 default_language)

    def as_skos_file(self):
        """return the vocabulary as a file object, in SKOS/Turtle syntax"""
        return open(os.path.join(self.datadir, 'subjects.ttl'), 'rb')

    def as_graph(self):
        """return the vocabulary as an rdflib graph"""
        return self.skos.graph
