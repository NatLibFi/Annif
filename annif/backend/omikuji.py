"""Annif backend using the Omikuji classifier"""

import omikuji
import os.path
import shutil
import annif.util
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend
from . import mixins


class OmikujiBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """Omikuji based backend for Annif"""
    name = "omikuji"
    needs_subject_index = True

    # defaults for uninitialized instances
    _model = None

    TRAIN_FILE = 'omikuji-train.txt'
    MODEL_FILE = 'omikuji-model'

    DEFAULT_PARAMETERS = {
        'min_df': 1,
        'cluster_balanced': True,
        'cluster_k': 2,
        'max_depth': 20,
        'collapse_every_n_layers': 0,
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
                self._model = omikuji.Model.load(path)
            else:
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)

    def initialize(self):
        self.initialize_vectorizer()
        self._initialize_model()

    def _uris_to_subj_ids(self, uris):
        subject_ids = [self.project.subjects.by_uri(uri)
                       for uri in uris]
        return [str(subj_id)
                for subj_id in subject_ids
                if subj_id is not None]

    def _create_train_file(self, veccorpus, corpus):
        self.info('creating train file')
        path = os.path.join(self.datadir, self.TRAIN_FILE)
        with open(path, 'w', encoding='utf-8') as trainfile:
            # Extreme Classification Repository format header line
            # We don't yet know the number of samples, as some may be skipped
            print('00000000',
                  len(self.vectorizer.vocabulary_),
                  len(self.project.subjects),
                  file=trainfile)
            n_samples = 0
            for doc, vector in zip(corpus.documents, veccorpus):
                subject_ids = self._uris_to_subj_ids(doc.uris)
                feature_values = ['{}:{}'.format(col, vector[row, col])
                                  for row, col in zip(*vector.nonzero())]
                if not subject_ids or not feature_values:
                    continue  # noqa
                print(','.join(subject_ids),
                      ' '.join(feature_values),
                      file=trainfile)
                n_samples += 1
            # replace the number of samples value at the beginning
            trainfile.seek(0)
            print('{:08d}'.format(n_samples), end='', file=trainfile)

    def _create_model(self, params):
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        model_path = os.path.join(self.datadir, self.MODEL_FILE)
        hyper_param = omikuji.Model.default_hyper_param()

        hyper_param.cluster_balanced = annif.util.boolean(
            params['cluster_balanced'])
        hyper_param.cluster_k = int(params['cluster_k'])
        hyper_param.max_depth = int(params['max_depth'])
        hyper_param.collapse_every_n_layers = int(
            params['collapse_every_n_layers'])

        self._model = omikuji.Model.train_on_data(train_path, hyper_param)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        self._model.save(os.path.join(self.datadir, self.MODEL_FILE))

    def _train(self, corpus, params):
        if corpus != 'cached':
            if corpus.is_empty():
                raise NotSupportedException(
                    'Cannot train omikuji project with no documents')
            input = (doc.text for doc in corpus.documents)
            vecparams = {'min_df': int(params['min_df']),
                         'tokenizer': self.project.analyzer.tokenize_words}
            veccorpus = self.create_vectorizer(input, vecparams)
            self._create_train_file(veccorpus, corpus)
        else:
            self.info("Reusing cached training data from previous run.")
        self._create_model(params)

    def _suggest(self, text, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        vector = self.vectorizer.transform([text])
        if vector.nnz == 0:  # All zero vector, empty result
            return ListSuggestionResult([])
        feature_values = [(col, vector[row, col])
                          for row, col in zip(*vector.nonzero())]
        results = []
        limit = int(params['limit'])
        for subj_id, score in self._model.predict(feature_values, top_k=limit):
            subject = self.project.subjects[subj_id]
            results.append(SubjectSuggestion(
                uri=subject[0],
                label=subject[1],
                notation=subject[2],
                score=score))
        return ListSuggestionResult(results)
