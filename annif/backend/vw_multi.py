"""Annif backend using the Vorpal Wabbit multiclass and multilabel
classifiers"""

import random
import os.path
import annif.util
from vowpalwabbit import pyvw
import numpy as np
from annif.hit import VectorAnalysisResult
from annif.exception import ConfigurationException, NotInitializedException
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

    DEFAULT_ALGORITHM = 'oaa'
    SUPPORTED_ALGORITHMS = ('oaa', 'ect')

    MODEL_FILE = 'vw-model'
    TRAIN_FILE = 'vw-train.txt'

    # defaults for uninitialized instances
    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self._get_datadir(), self.MODEL_FILE)
            if not os.path.exists(path):
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)
            self.debug('loading VW model from {}'.format(path))
            params = self._create_params({'i': path, 'quiet': True})
            self._model = pyvw.vw(**params)
            self.debug('loaded model {}'.format(str(self._model)))

    @property
    def algorithm(self):
        algorithm = self.params.get('algorithm', self.DEFAULT_ALGORITHM)
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ConfigurationException(
                "{} is not a valid algorithm (allowed: {})".format(
                    algorithm, ', '.join(self.SUPPORTED_ALGORITHMS)),
                backend_id=self.backend_id)
        return algorithm

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

    def _create_params(self, params):
        params.update({param: defaultval
                       for param, (_, defaultval) in self.VW_PARAMS.items()
                       if defaultval is not None})
        params.update({param: self._convert_param(param, val)
                       for param, val in self.params.items()
                       if param in self.VW_PARAMS})
        if params.get('passes', 1) > 1:
            # need a cache file when there are multiple passes
            params['cache'] = True
            params['kill_cache'] = True
        if self.algorithm == 'oaa':
            # only the oaa algorithm supports probabilities output
            params['probabilities'] = True
            params['loss_function'] = 'logistic'
        return params

    def _create_model(self, project):
        self.info('creating VW model (algorithm: {})'.format(self.algorithm))
        trainpath = os.path.join(self._get_datadir(), self.TRAIN_FILE)
        params = self._create_params(
            {'data': trainpath, self.algorithm: len(project.subjects)})
        self.debug("model parameters: {}".format(params))
        self._model = pyvw.vw(**params)
        modelpath = os.path.join(self._get_datadir(), self.MODEL_FILE)
        self._model.save(modelpath)

    def train(self, corpus, project):
        self._create_train_file(corpus, project)
        self._create_model(project)

    def _analyze_chunks(self, chunktexts, project):
        results = []
        for chunktext in chunktexts:
            example = ' | {}'.format(chunktext)
            result = self._model.predict(example)
            if isinstance(result, int):
                # just a single integer - need to one-hot-encode
                mask = np.zeros(len(project.subjects))
                mask[result - 1] = 1.0
                result = mask
            else:
                result = np.array(result)
            results.append(result)
        return VectorAnalysisResult(
            np.array(results).mean(axis=0), project.subjects)
