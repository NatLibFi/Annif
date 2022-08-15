"""Annif backend using a SVM classifier"""

import os.path
import joblib
import numpy as np
import scipy.special
from sklearn.svm import LinearSVC
import annif.util
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend
from . import mixins


class SVCBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """Support vector classifier backend for Annif"""
    name = "svc"
    needs_subject_index = True

    # defaults for uninitialized instances
    _model = None

    MODEL_FILE = 'svc-model.gz'

    DEFAULT_PARAMETERS = {
        'min_df': 1,
        'ngram': 1
    }

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _initialize_model(self):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug('loading model from {}'.format(path))
            if os.path.exists(path):
                self._model = joblib.load(path)
            else:
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)

    def initialize(self, parallel=False):
        self.initialize_vectorizer()
        self._initialize_model()

    def _corpus_to_texts_and_classes(self, corpus):
        texts = []
        classes = []
        for doc in corpus.documents:
            texts.append(doc.text)
            if len(doc.uris) > 1:
                self.warning(
                    'training on a document with multiple subjects is not ' +
                    'supported by SVC; selecting one random subject.')
            classes.append(next(iter(doc.uris)))
        return texts, classes

    def _train_classifier(self, veccorpus, classes):
        self.info('creating classifier')
        self._model = LinearSVC()
        self._model.fit(veccorpus, classes)
        annif.util.atomic_save(self._model,
                               self.datadir,
                               self.MODEL_FILE,
                               method=joblib.dump)

    def _train(self, corpus, params, jobs=0):
        if corpus == 'cached':
            raise NotSupportedException(
                'SVC backend does not support reuse of cached training data.')
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train SVC project with no documents')
        texts, classes = self._corpus_to_texts_and_classes(corpus)
        vecparams = {'min_df': int(params['min_df']),
                     'tokenizer': self.project.analyzer.tokenize_words,
                     'ngram_range': (1, int(params['ngram']))}
        veccorpus = self.create_vectorizer(texts, vecparams)
        self._train_classifier(veccorpus, classes)

    def _scores_to_suggestions(self, scores, params):
        results = []
        limit = int(params['limit'])
        for class_id in np.argsort(scores)[::-1][:limit]:
            class_uri = self._model.classes_[class_id]
            subject_id = self.project.subjects.by_uri(class_uri)
            if subject_id is not None:
                uri, label, notation = self.project.subjects[subject_id]
                results.append(SubjectSuggestion(
                    uri=uri,
                    label=label,
                    notation=notation,
                    score=scores[class_id]))
        return ListSuggestionResult(results)

    def _suggest(self, text, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        vector = self.vectorizer.transform([text])
        if vector.nnz == 0:  # All zero vector, empty result
            return ListSuggestionResult([])
        confidences = self._model.decision_function(vector)[0]
        # convert to 0..1 score range using logistic function
        scores = scipy.special.expit(confidences)
        return self._scores_to_suggestions(scores, params)
