"""Backend that returns most similar subjects based on doc2vec similarity"""

import os.path
import gensim.models.doc2vec
import annif.util
from annif.hit import AnalysisHit, ListAnalysisResult
from annif.exception import NotInitializedException
from . import backend


class Doc2VecBackend(backend.AnnifBackend):
    """Doc2Vec similarity based backend for Annif"""
    name = "doc2vec"
    needs_subject_index = True

    # defaults for uninitialized instances
    _model = None

    MODEL_FILE = 'doc2vec-model'

    def initialize(self):
        if self._model is None:
            path = os.path.join(self._get_datadir(), self.MODEL_FILE)
            self.debug('loading model from {}'.format(path))
            if os.path.exists(path):
                self._model = gensim.models.doc2vec.Doc2Vec.load(path)
            else:
                raise NotInitializedException(
                    'model file {} not found'.format(path),
                    backend_id=self.backend_id)

    @classmethod
    def _build_corpus(cls, corpus, project):
        for subject_id, subj in enumerate(corpus.subjects):
            words = project.analyzer.tokenize_words(subj.text)
            tags = [subject_id]
            yield gensim.models.doc2vec.TaggedDocument(words=words, tags=tags)

    def train(self, corpus, project):
        self.info('creating doc2vec model')
        self._model = gensim.models.doc2vec.Doc2Vec(
            vector_size=50, min_count=2, epochs=10)
        self._model.build_vocab(self._build_corpus(corpus, project))
        self._model.train(
            self._build_corpus(
                corpus,
                project),
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs)

        annif.util.atomic_save(
            self._model,
            self._get_datadir(),
            self.MODEL_FILE)

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        words = project.analyzer.tokenize_words(text)
        vector = self._model.infer_vector(words)
        limit = int(self.params['limit'])
        sims = self._model.docvecs.most_similar([vector], topn=limit)
        results = []
        for subject_id, score in sims:
            subj = project.subjects[subject_id]
            results.append(
                AnalysisHit(
                    uri=subj[0],
                    label=subj[1],
                    score=score))
        return ListAnalysisResult(results, project.subjects)
