"""Annif backend using the Vorpal Wabbit multiclass and multilabel
classifiers"""

import random
import os.path
import annif.util
from vowpalwabbit import pyvw
import numpy as np
from annif.hit import ListAnalysisResult, VectorAnalysisResult
from annif.exception import ConfigurationException, NotInitializedException
from . import backend
from . import mixins


class VWMultiBackend(mixins.ChunkingBackend, backend.AnnifLearningBackend):
    """Vorpal Wabbit multiclass/multilabel backend for Annif"""

    name = "vw_multi"
    needs_subject_index = True

    VW_PARAMS = {
        # each param specifier is a pair (allowed_values, default_value)
        # where allowed_values is either a type or a list of allowed values
        # and default_value may be None, to let VW decide by itself
        'bit_precision': (int, None),
        'ngram': (lambda x: '_{}'.format(int(x)), None),
        'learning_rate': (float, None),
        'loss_function': (['squared', 'logistic', 'hinge'], 'logistic'),
        'l1': (float, None),
        'l2': (float, None),
        'passes': (int, None),
        'probabilities': (bool, None)
    }

    DEFAULT_ALGORITHM = 'oaa'
    SUPPORTED_ALGORITHMS = ('oaa', 'ect', 'log_multi', 'multilabel_oaa')

    DEFAULT_INPUTS = '_text_'

    MODEL_FILE = 'vw-model'
    TRAIN_FILE = 'vw-train.txt'

    # defaults for uninitialized instances
    _model = None

    def initialize(self):
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

    @property
    def algorithm(self):
        algorithm = self.params.get('algorithm', self.DEFAULT_ALGORITHM)
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

    @staticmethod
    def _normalize_text(project, text):
        ntext = ' '.join(project.analyzer.tokenize_words(text))
        return VWMultiBackend._cleanup_text(ntext)

    @staticmethod
    def _write_train_file(examples, filename):
        with open(filename, 'w') as trainfile:
            for ex in examples:
                print(ex, file=trainfile)

    @staticmethod
    def _uris_to_subject_ids(project, uris):
        subject_ids = []
        for uri in uris:
            subject_id = project.subjects.by_uri(uri)
            if subject_id is not None:
                subject_ids.append(subject_id)
        return subject_ids

    def _format_examples(self, project, text, uris):
        subject_ids = self._uris_to_subject_ids(project, uris)
        if self.algorithm == 'multilabel_oaa':
            yield '{} {}'.format(','.join(map(str, subject_ids)), text)
        else:
            for subject_id in subject_ids:
                yield '{} {}'.format(subject_id + 1, text)

    def _get_input(self, input, project, text):
        if input == '_text_':
            return self._normalize_text(project, text)
        else:
            proj = annif.project.get_project(input)
            result = proj.analyze(text)
            features = [
                '{}:{}'.format(self._cleanup_text(hit.uri), hit.score)
                for hit in result.hits]
            return ' '.join(features)

    def _inputs_to_exampletext(self, project, text):
        namespaces = {}
        for input in self.inputs:
            inputtext = self._get_input(input, project, text)
            if inputtext:
                namespaces[input] = inputtext
        if not namespaces:
            return None
        return ' '.join(['|{} {}'.format(namespace, featurestr)
                         for namespace, featurestr in namespaces.items()])

    def _create_examples(self, corpus, project):
        examples = []
        for doc in corpus.documents:
            text = self._inputs_to_exampletext(project, doc.text)
            if not text:
                continue
            examples.extend(self._format_examples(project, text, doc.uris))
        random.shuffle(examples)
        return examples

    def _create_train_file(self, corpus, project):
        self.info('creating VW train file')
        examples = self._create_examples(corpus, project)
        annif.util.atomic_save(examples,
                               self.datadir,
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
        return params

    def _create_model(self, project):
        self.info('creating VW model (algorithm: {})'.format(self.algorithm))
        trainpath = os.path.join(self.datadir, self.TRAIN_FILE)
        params = self._create_params(
            {'data': trainpath, self.algorithm: len(project.subjects)})
        if params.get('passes', 1) > 1:
            # need a cache file when there are multiple passes
            params.update({'cache': True, 'kill_cache': True})
        self.debug("model parameters: {}".format(params))
        self._model = pyvw.vw(**params)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)

    def train(self, corpus, project):
        self._create_train_file(corpus, project)
        self._create_model(project)

    def learn(self, corpus, project):
        for example in self._create_examples(corpus, project):
            self._model.learn(example)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)

    def _convert_result(self, result, project):
        if self.algorithm == 'multilabel_oaa':
            # result is a list of subject IDs - need to vectorize
            mask = np.zeros(len(project.subjects))
            mask[result] = 1.0
            return mask
        elif isinstance(result, int):
            # result is a single integer - need to one-hot-encode
            mask = np.zeros(len(project.subjects))
            mask[result - 1] = 1.0
            return mask
        else:
            # result is a list of scores (probabilities or binary 1/0)
            return np.array(result)

    def _analyze_chunks(self, chunktexts, project):
        results = []
        for chunktext in chunktexts:
            exampletext = self._inputs_to_exampletext(project, chunktext)
            if not exampletext:
                continue
            example = ' {}'.format(exampletext)
            result = self._model.predict(example)
            results.append(self._convert_result(result, project))
        if not results:  # empty result
            return ListAnalysisResult(hits=[], subject_index=project.subjects)
        return VectorAnalysisResult(
            np.array(results).mean(axis=0), project.subjects)
