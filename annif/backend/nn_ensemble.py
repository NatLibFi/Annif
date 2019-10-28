"""Neural network based ensemble backend that combines results from multiple
projects."""


import os.path
import numpy as np
from tensorflow.keras.layers import Input, Dense, Add, Flatten, Lambda, Dropout
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import annif.corpus
import annif.project
import annif.util
from annif.exception import NotInitializedException
from annif.suggestion import VectorSuggestionResult
from . import ensemble


class NNEnsembleBackend(ensemble.EnsembleBackend):
    """Neural network ensemble backend that combines results from multiple
    projects"""

    name = "nn_ensemble"

    MODEL_FILE = "nn-model.h5"

    DEFAULT_PARAMS = {
        'nodes': 100,
        'dropout_rate': 0.2,
        'optimizer': 'adam',
        'epochs': 10,
    }

    # defaults for uninitialized instances
    _model = None

    def default_params(self):
        params = {}
        params.update(super().default_params())
        params.update(self.DEFAULT_PARAMS)
        return params

    def initialize(self):
        if self._model is not None:
            return  # already initialized
        model_filename = os.path.join(self.datadir, self.MODEL_FILE)
        if not os.path.exists(model_filename):
            raise NotInitializedException(
                'model file {} not found'.format(model_filename),
                backend_id=self.backend_id)
        self.debug('loading Keras model from {}'.format(model_filename))
        self._model = load_model(model_filename)

    def _merge_hits_from_sources(self, hits_from_sources, project, params):
        score_vector = np.array([hits.vector * weight
                                 for hits, weight in hits_from_sources],
                                dtype=np.float32)
        results = self._model.predict(
            np.expand_dims(score_vector.transpose(), 0))
        return VectorSuggestionResult(results[0], project.subjects)

    def _create_model(self, sources, project):
        self.info("creating NN ensemble model")

        inputs = Input(shape=(len(project.subjects), len(sources)))

        flat_input = Flatten()(inputs)
        drop_input = Dropout(
            rate=float(
                self.params['dropout_rate']))(flat_input)
        hidden = Dense(int(self.params['nodes']),
                       activation="relu")(drop_input)
        drop_hidden = Dropout(rate=float(self.params['dropout_rate']))(hidden)
        delta = Dense(len(project.subjects),
                      kernel_initializer='zeros',
                      bias_initializer='zeros')(drop_hidden)

        mean = Lambda(lambda x: K.mean(x, axis=2))(inputs)

        predictions = Add()([mean, delta])

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.compile(optimizer=self.params['optimizer'],
                            loss='binary_crossentropy',
                            metrics=['top_k_categorical_accuracy'])

        summary = []
        self._model.summary(print_fn=summary.append)
        self.debug("Created model: \n" + "\n".join(summary))

    def train(self, corpus, project):
        sources = annif.util.parse_sources(self.params['sources'])
        self._create_model(sources, project)
        self.learn(corpus, project)

    def _corpus_to_vectors(self, corpus, project):
        # pass corpus through all source projects
        sources = [(annif.project.get_project(project_id), weight)
                   for project_id, weight
                   in annif.util.parse_sources(self.params['sources'])]

        score_vectors = []
        true_vectors = []
        for doc in corpus.documents:
            doc_scores = []
            for source_project, weight in sources:
                hits = source_project.suggest(doc.text)
                doc_scores.append(hits.vector * weight)
            score_vectors.append(np.array(doc_scores,
                                          dtype=np.float32).transpose())
            subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
            true_vectors.append(subjects.as_vector(project.subjects))
        # collect the results into a single vector, considering weights
        scores = np.array(score_vectors, dtype=np.float32)
        # collect the gold standard values into another vector
        true = np.array(true_vectors, dtype=np.float32)
        return (scores, true)

    def learn(self, corpus, project):
        scores, true = self._corpus_to_vectors(corpus, project)

        # fit the model
        self._model.fit(scores, true, batch_size=32, verbose=True,
                        epochs=int(self.params['epochs']))

        annif.util.atomic_save(
            self._model,
            self.datadir,
            self.MODEL_FILE)
