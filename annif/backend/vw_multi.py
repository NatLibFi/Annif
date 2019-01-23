"""Annif backend using the Vorpal Wabbit multiclass and multilabel
classifiers"""

import random
import os.path
import annif.util
from vowpalwabbit import pyvw
import numpy as np
from annif.hit import AnalysisHit, VectorAnalysisResult
from annif.exception import NotInitializedException
from . import backend
from . import mixins


class VWMultiBackend(mixins.ChunkingBackend, backend.AnnifBackend):
    """Vorpal Wabbit multiclass/multilabel backend for Annif"""

    name = "vw_multi"
    needs_subject_index = True

    VW_PARAMS = {
        # each param specifier is a pair (allowed_values, default_value)
        # where allowed_values is either a type or a list of allowed values
        # and default_value may be None, to let VW decide by itself
        'bit_precision': (int, None),
        'learning_rate': (float, None),
        'loss_function': (['squared', 'logistic', 'hinge'], 'logistic'),
        'l1': (float, None),
        'l2': (float, None),
        'passes': (int, None)
    }

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
        ntext = ' '.join(project.analyzer.tokenize_words(text))
        # colon and pipe chars have special meaning in VW and must be avoided
        return ntext.replace(':', '').replace('|', '')

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

    def _convert_param(self, param, val):
        pspec, _ = self.VW_PARAMS[param]
        if isinstance(pspec, list):
            if val in pspec:
                return val
            raise ConfigurationException(
                "{} is not a valid value for {} (allowed: {})".format(
                    val, param, ', '.join(pspec)), backend_id=self.backend_id)
        try:
            return pspec(val)
        except ValueError:
            raise ConfigurationException(
                "The {} value {} cannot be converted to {}".format(
                    param, val, pspec), backend_id=self.backend_id)

    def _create_model(self, project):
        self.info('creating VW model')
        trainpath = os.path.join(self._get_datadir(), self.TRAIN_FILE)
        params = {param: defaultval
                  for param, (_, defaultval) in self.VW_PARAMS.items()
                  if defaultval is not None}
        params.update({param: self._convert_param(param, val)
                       for param, val in self.params.items()
                       if param in self.VW_PARAMS})
        self.debug("model parameters: {}".format(params))
        self._model = pyvw.vw(
            oaa=len(project.subjects),
            probabilities=True,
            data=trainpath,
            **params)
        modelpath = os.path.join(self._get_datadir(), self.MODEL_FILE)
        self._model.save(modelpath)

    def train(self, corpus, project):
        self._create_train_file(corpus, project)
        self._create_model(project)

    def _analyze_chunks(self, chunktexts, project):
        results = []
        for chunktext in chunktexts:
            example = ' | {}'.format(chunktext)
            results.append(np.array(self._model.predict(example)))
        return VectorAnalysisResult(
            np.array(results).mean(axis=0), project.subjects)
