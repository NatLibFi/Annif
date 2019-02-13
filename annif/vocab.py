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

    def load_vocabulary(self, subject_corpus):
        """load subjects from a subject corpus"""

        self._create_subject_index(subject_corpus)
