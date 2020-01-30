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
        updated_subjects = annif.corpus.SubjectIndex.load(os.path.devnull)

        for subject_id in range(len(old_subjects)):
            old_uri = old_subjects[subject_id][0]
            new_ind = new_subjects.by_uri(old_uri)
            if new_ind is None:  # subject removed from new corpus
                new_label = ''
            else:
                new_label = new_subjects[new_ind][1]
            updated_subjects.append(old_uri, new_label)

        for subject_id in range(len(new_subjects)):
            new_uri = new_subjects[subject_id][0]
            new_label = new_subjects[subject_id][1]
            if new_uri in old_subjects._uris:
                continue
            updated_subjects.append(new_uri, new_label)

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
