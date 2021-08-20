"""Annif backend using the Vowpal Wabbit multiclass and multilabel
classifiers"""

import os
import random
import numpy as np
from vowpalwabbit import pyvw
import annif.util
from annif.suggestion import ListSuggestionResult, VectorSuggestionResult
from annif.exception import ConfigurationException
from annif.exception import NotInitializedException
from . import backend
from . import mixins


class VWMultiBackend(mixins.ChunkingBackend, backend.AnnifLearningBackend):
    """Vowpal Wabbit multiclass/multilabel backend for Annif"""

    name = "vw_multi"
    needs_subject_index = True

    MODEL_FILE = 'vw-model'
    TRAIN_FILE = 'vw-train.txt'

    # defaults for uninitialized instances
    _model = None

    VW_PARAMS = {
        'bit_precision': (int, None),
        'ngram': (lambda x: '_{}'.format(int(x)), None),
        'learning_rate': (float, None),
        'loss_function': (['squared', 'logistic', 'hinge'], 'logistic'),
        'l1': (float, None),
        'l2': (float, None),
        'passes': (int, None),
        'probabilities': (bool, None),
        'quiet': (bool, False),
        'data': (str, None),
        'i': (str, None),
    }

    SUPPORTED_ALGORITHMS = ('oaa', 'ect', 'log_multi', 'multilabel_oaa')
    VW_PARAMS.update({alg: (int, None) for alg in SUPPORTED_ALGORITHMS})

    DEFAULT_INPUTS = '_text_'

    DEFAULT_PARAMETERS = {'algorithm': 'oaa'}

    def initialize(self, parallel=False):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            if not os.path.exists(path):
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)
            self.debug('loading VW model from {}'.format(path))
            params = self._create_params({'i': path, 'quiet': True})
            if 'passes' in params:
                # don't confuse the model with passes
                del params['passes']
            self.debug("model parameters: {}".format(params))
            self._model = pyvw.vw(**params)
            self.debug('loaded model {}'.format(str(self._model)))

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

    def _create_params(self, initial_params):
        params = {param: self._convert_param(param, val)
                  for param, val in self.params.items()
                  if param in self.VW_PARAMS}
        params.update({param: self._convert_param(param, val)
                       for param, val in initial_params.items()
                       if param in self.VW_PARAMS})
        return params

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(mixins.ChunkingBackend.DEFAULT_PARAMETERS)
        params.update(self.DEFAULT_PARAMETERS)
        params.update({param: default_val
                       for param, (_, default_val) in self.VW_PARAMS.items()
                       if default_val is not None})
        return params

    @property
    def algorithm(self):
        algorithm = self.params['algorithm']
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ConfigurationException(
                "{} is not a valid algorithm (allowed: {})".format(
                    algorithm, ', '.join(self.SUPPORTED_ALGORITHMS)),
                backend_id=self.backend_id)
        return algorithm

    @property
    def inputs(self):
        inputs = self.params.get('inputs', self.DEFAULT_INPUTS)
        return inputs.split(',')

    @staticmethod
    def _cleanup_text(text):
        # colon and pipe chars have special meaning in VW and must be avoided
        return text.replace(':', '').replace('|', '')

    def _normalize_text(self, text):
        ntext = ' '.join(self.project.analyzer.tokenize_words(text))
        return VWMultiBackend._cleanup_text(ntext)

    def _uris_to_subject_ids(self, uris):
        subject_ids = []
        for uri in uris:
            subject_id = self.project.subjects.by_uri(uri)
            if subject_id is not None:
                subject_ids.append(subject_id)
        return subject_ids

    def _format_examples(self, text, uris):
        subject_ids = self._uris_to_subject_ids(uris)
        if self.algorithm == 'multilabel_oaa':
            yield '{} {}'.format(','.join(map(str, subject_ids)), text)
        else:
            for subject_id in subject_ids:
                yield '{} {}'.format(subject_id + 1, text)

    def _get_input(self, input, text):
        if input == '_text_':
            return self._normalize_text(text)
        else:
            proj = self.project.registry.get_project(input)
            result = proj.suggest(text)
            features = [
                '{}:{}'.format(self._cleanup_text(hit.uri), hit.score)
                for hit in result.as_list(self.project.subjects)]
            return ' '.join(features)

    def _inputs_to_exampletext(self, text):
        namespaces = {}
        for input in self.inputs:
            inputtext = self._get_input(input, text)
            if inputtext:
                namespaces[input] = inputtext
        if not namespaces:
            return None
        return ' '.join(['|{} {}'.format(namespace, featurestr)
                         for namespace, featurestr in namespaces.items()])

    def _create_examples(self, corpus):
        examples = []
        for doc in corpus.documents:
            text = self._inputs_to_exampletext(doc.text)
            if not text:
                continue
            examples.extend(self._format_examples(text, doc.uris))
        random.shuffle(examples)
        return examples

    def _create_model(self, params):
        self.info('creating VW model (algorithm: {})'.format(self.algorithm))
        trainpath = os.path.join(self.datadir, self.TRAIN_FILE)
        initial_params = {'data': trainpath,
                          self.algorithm: len(self.project.subjects),
                          **{key: val for key, val in params.items()
                              if key in self.VW_PARAMS}}
        params = self._create_params(initial_params)
        if params.get('passes', 1) > 1:
            # need a cache file when there are multiple passes
            params.update({'cache': True, 'kill_cache': True})
        self.debug("model parameters: {}".format(params))
        self._model = pyvw.vw(**params)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)

    def _convert_result(self, result):
        if self.algorithm == 'multilabel_oaa':
            # result is a list of subject IDs - need to vectorize
            mask = np.zeros(len(self.project.subjects), dtype=np.float32)
            mask[result] = 1.0
            return mask
        elif isinstance(result, int):
            # result is a single integer - need to one-hot-encode
            mask = np.zeros(len(self.project.subjects), dtype=np.float32)
            mask[result - 1] = 1.0
            return mask
        else:
            # result is a list of scores (probabilities or binary 1/0)
            return np.array(result, dtype=np.float32)

    def _suggest_chunks(self, chunktexts, params):
        results = []
        for chunktext in chunktexts:

            exampletext = self._inputs_to_exampletext(chunktext)
            if not exampletext:
                continue
            example = ' {}'.format(exampletext)
            result = self._model.predict(example)
            results.append(self._convert_result(result))
        if not results:  # empty result
            return ListSuggestionResult([])
        return VectorSuggestionResult(
            np.array(results, dtype=np.float32).mean(axis=0))

    @staticmethod
    def _write_train_file(examples, filename):
        with open(filename, 'w', encoding='utf-8') as trainfile:
            for ex in examples:
                print(ex, file=trainfile)

    def _create_train_file(self, corpus):
        self.info('creating VW train file')
        examples = self._create_examples(corpus)
        annif.util.atomic_save(examples,
                               self.datadir,
                               self.TRAIN_FILE,
                               method=self._write_train_file)

    def _train(self, corpus, params, jobs=0):
        if corpus != 'cached':
            self._create_train_file(corpus)
        else:
            self.info("Reusing cached training data from previous run.")
        self._create_model(params)

    def _learn(self, corpus, params):
        self.initialize()
        for example in self._create_examples(corpus):
            self._model.learn(example)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)
