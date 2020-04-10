"""Vocabulary management functionality for Annif"""

import os.path
import annif
import annif.corpus
import annif.util
from annif.datadir import DatadirMixin
from annif.exception import NotInitializedException

logger = annif.logger


class AnnifVocabulary(DatadirMixin):
    """Class representing a subject vocabulary which can be used by multiple
    Annif projects."""

    # defaults for uninitialized instances
    _subjects = None

    def __init__(self, vocab_id, datadir):
        DatadirMixin.__init__(self, datadir, 'vocabs', vocab_id)
        self.vocab_id = vocab_id

    def _create_subject_index(self, subject_corpus):
        self._subjects = annif.corpus.SubjectIndex(subject_corpus)
        annif.util.atomic_save(self._subjects, self.datadir, 'subjects')

    def _update_subject_index(self, subject_corpus):
        old_subjects = self.subjects
        new_subjects = annif.corpus.SubjectIndex(subject_corpus)
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
        self._subjects = updated_subjects
        annif.util.atomic_save(self._subjects, self.datadir, 'subjects')

    @property
    def subjects(self):
        if self._subjects is None:
            path = os.path.join(self.datadir, 'subjects')
            if os.path.exists(path):
                logger.debug('loading subjects from %s', path)
                self._subjects = annif.corpus.SubjectIndex.load(path)
            else:
                raise NotInitializedException(
                    "subject file {} not found".format(path))
        return self._subjects

    def load_vocabulary(self, subject_corpus, language):
        """load subjects from a subject corpus and save them into a
        SKOS/Turtle file for later use"""

        if os.path.exists(os.path.join(self.datadir, 'subjects')):
            logger.info('updating existing vocabulary')
            self._update_subject_index(subject_corpus)
        else:
            self._create_subject_index(subject_corpus)
        subject_corpus.save_skos(os.path.join(self.datadir, 'subjects.ttl'),
                                 language)

    def as_skos(self):
        """return the vocabulary as a file object, in SKOS/Turtle syntax"""
        return open(os.path.join(self.datadir, 'subjects.ttl'), 'rb')
