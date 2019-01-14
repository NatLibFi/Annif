"""Annif backend using the Vorpal Wabbit classifier"""

import random
import os.path
import annif.util
from vowpalwabbit import pyvw
import numpy as np
from annif.hit import AnalysisHit, VectorAnalysisResult
from annif.exception import NotInitializedException
from . import backend


class VWBackend(backend.AnnifBackend):
    """Vorpal Wabbit backend for Annif"""

    name = "vw"
    needs_subject_index = True

    MODEL_FILE = 'vw-model'
    TRAIN_FILE = 'vw-train.txt'

    # defaults for uninitialized instances
    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self._get_datadir(), self.MODEL_FILE)
            self.debug('loading VW model from {}'.format(path))
            if os.path.exists(path):
                self._model = pyvw.vw(
                    i=path,
                    quiet=True,
                    loss_function='logistic',
                    probabilities=True)
                self.debug('loaded model {}'.format(str(self._model)))
            else:
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)

    @classmethod
    def _label_to_subject(cls, project, label):
        subject_id = int(label) - 1
        return project.subjects[subject_id]

    @classmethod
    def _normalize_text(cls, project, text):
        return ' '.join(project.analyzer.tokenize_words(text)).replace(':', '')

    def _write_train_file(self, examples, filename):
        with open(filename, 'w') as trainfile:
            for ex in examples:
                print(ex, file=trainfile)

    def _create_train_file(self, corpus, project):
        self.info('creating VW train file')
        examples = []
        for doc in corpus.documents:
            text = self._normalize_text(project, doc.text)
            for uri in doc.uris:
                subject_id = project.subjects.by_uri(uri)
                if subject_id is None:
                    continue
                exstr = '{} | {}'.format(subject_id + 1, text)
                examples.append(exstr)
        random.shuffle(examples)
        annif.util.atomic_save(examples,
                               self._get_datadir(),
                               self.TRAIN_FILE,
                               method=self._write_train_file)

    def _create_model(self, project):
        self.info('creating VW model')
        trainpath = os.path.join(self._get_datadir(), self.TRAIN_FILE)
        self._model = pyvw.vw(
            oaa=len(
                project.subjects),
            loss_function='logistic',
            probabilities=True,
            data=trainpath,
            b=28)
        modelpath = os.path.join(self._get_datadir(), self.MODEL_FILE)
        self._model.save(modelpath)

    def load_corpus(self, corpus, project):
        self._create_train_file(corpus, project)
        self._create_model(project)

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        normalized = self._normalize_text(project, text)
        example = ' | {}'.format(normalized)
        result = self._model.predict(example)
        return VectorAnalysisResult(np.array(result), project.subjects)
