"""Annif backend using the Omikuji classifier"""

import omikuji
import os.path
import shutil
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import annif.util
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend


class OmikujiBackend(backend.AnnifBackend):
    """Omikuji based backend for Annif"""
    name = "omikuji"
    needs_subject_index = True

    # defaults for uninitialized instances
    _vectorizer = None
    _model = None

    VECTORIZER_FILE = 'vectorizer'
    TRAIN_FILE = 'omikuji-train.txt'
    MODEL_FILE = 'omikuji-model'

    def _initialize_vectorizer(self):
        if self._vectorizer is None:
            path = os.path.join(self.datadir, self.VECTORIZER_FILE)
            if os.path.exists(path):
                self.debug('loading vectorizer from {}'.format(path))
                self._vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    backend_id=self.backend_id)

    def _initialize_model(self):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug('loading model from {}'.format(path))
            if os.path.exists(path):
                self._model = omikuji.Model.load(path)
            else:
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)

    def initialize(self):
        self._initialize_vectorizer()
        self._initialize_model()

    def _create_train_file(self, veccorpus, corpus):
        self.info('creating train file')
        path = os.path.join(self.datadir, self.TRAIN_FILE)
        with open(path, 'w', encoding='utf-8') as trainfile:
            # Extreme Classification Repository format header line
            # We don't yet know the number of samples, as some may be skipped
            print('00000000',
                  len(self._vectorizer.vocabulary_),
                  len(self.project.subjects),
                  file=trainfile)
            n_samples = 0
            for doc, vector in zip(corpus.documents, veccorpus):
                subject_ids = [self.project.subjects.by_uri(uri)
                               for uri in doc.uris]
                subject_id_str = [str(subj_id)
                                  for subj_id in subject_ids
                                  if subj_id is not None]
                feature_values = ['{}:{}'.format(col, vector[row, col])
                                  for row, col in zip(*vector.nonzero())]
                if not subject_id_str or not feature_values:
                    continue
                print(','.join(subject_id_str),
                      ' '.join(feature_values),
                      file=trainfile)
                n_samples += 1
            # replace the number of samples value at the beginning
            trainfile.seek(0)
            print('{:08d}'.format(n_samples), end='', file=trainfile)

    def _create_model(self):
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        model_path = os.path.join(self.datadir, self.MODEL_FILE)
        hyper_param = omikuji.Model.default_hyper_param()

        # Bonsai hyperparameters
        hyper_param.cluster_balanced = False
        hyper_param.cluster_k = 100
        hyper_param.max_depth = 3

        self._model = omikuji.Model.train_on_data(train_path, hyper_param)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        self._model.save(os.path.join(self.datadir, self.MODEL_FILE))

    def train(self, corpus):
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train omikuji project with no documents')
        self.info('creating vectorizer')
        self._vectorizer = TfidfVectorizer(
            #            min_df=5,
            tokenizer=self.project.analyzer.tokenize_words)
        veccorpus = self._vectorizer.fit_transform(
            (doc.text for doc in corpus.documents))
        annif.util.atomic_save(
            self._vectorizer,
            self.datadir,
            self.VECTORIZER_FILE,
            method=joblib.dump)
        self._create_train_file(veccorpus, corpus)
        self._create_model()

    def _suggest(self, text, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        vector = self._vectorizer.transform([text])
        feature_values = [(col, vector[row, col])
                          for row, col in zip(*vector.nonzero())]
        results = []
        for subj_id, score in self._model.predict(feature_values):
            subject = self.project.subjects[subj_id]
            results.append(SubjectSuggestion(
                uri=subject[0],
                label=subject[1],
                score=score))
        return ListSuggestionResult(results, self.project.subjects)
