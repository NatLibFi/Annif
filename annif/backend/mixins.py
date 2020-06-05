"""Annif backend mixins that can be used to implement features"""


import abc
import os.path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import annif.util
from annif.exception import NotInitializedException
from annif.suggestion import ListSuggestionResult


class ChunkingBackend(metaclass=abc.ABCMeta):
    """Annif backend mixin that implements chunking of input"""

    DEFAULT_PARAMETERS = {'chunksize': 1}

    def default_params(self):
        return self.DEFAULT_PARAMETERS

    @abc.abstractmethod
    def _suggest_chunks(self, chunktexts, params):
        """Suggest subjects for the chunked text; should be implemented by
        the subclass inheriting this mixin"""

        pass  # pragma: no cover

    def _suggest(self, text, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        sentences = self.project.analyzer.tokenize_sentences(text)
        self.debug('Found {} sentences'.format(len(sentences)))
        chunksize = int(params['chunksize'])
        chunktexts = []
        for i in range(0, len(sentences), chunksize):
            chunktexts.append(' '.join(sentences[i:i + chunksize]))
        self.debug('Split sentences into {} chunks'.format(len(chunktexts)))
        if len(chunktexts) == 0:  # no input, empty result
            return ListSuggestionResult([])
        return self._suggest_chunks(chunktexts, params)


class TfidfVectorizerMixin:
    """Annif backend mixin that implements TfidfVectorizer functionality"""

    VECTORIZER_FILE = 'vectorizer'

    vectorizer = None

    def initialize_vectorizer(self):
        if self.vectorizer is None:
            path = os.path.join(self.datadir, self.VECTORIZER_FILE)
            if os.path.exists(path):
                self.debug('loading vectorizer from {}'.format(path))
                self.vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    backend_id=self.backend_id)

    def create_vectorizer(self, input, params={}):
        self.info('creating vectorizer')
        self.vectorizer = TfidfVectorizer(**params)
        veccorpus = self.vectorizer.fit_transform(input)
        annif.util.atomic_save(
            self.vectorizer,
            self.datadir,
            self.VECTORIZER_FILE,
            method=joblib.dump)
        return veccorpus
