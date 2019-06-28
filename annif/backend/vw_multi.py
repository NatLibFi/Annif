"""Annif backend using the Vowpal Wabbit multiclass and multilabel
classifiers"""

import random
import numpy as np
import annif.project
from annif.suggestion import ListSuggestionResult, VectorSuggestionResult
from annif.exception import ConfigurationException
from . import vw_base
from . import mixins


class VWMultiBackend(mixins.ChunkingBackend, vw_base.VWBaseBackend):
    """Vowpal Wabbit multiclass/multilabel backend for Annif"""

    name = "vw_multi"
    needs_subject_index = True

    VW_PARAMS = {
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
            result = proj.suggest(text)
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

    def _create_model(self, project):
        self.info('creating VW model (algorithm: {})'.format(self.algorithm))
        super()._create_model(project, {self.algorithm: len(project.subjects)})

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

    def _suggest_chunks(self, chunktexts, project):
        results = []
        for chunktext in chunktexts:
            exampletext = self._inputs_to_exampletext(project, chunktext)
            if not exampletext:
                continue
            example = ' {}'.format(exampletext)
            result = self._model.predict(example)
            results.append(self._convert_result(result, project))
        if not results:  # empty result
            return ListSuggestionResult(
                hits=[], subject_index=project.subjects)
        return VectorSuggestionResult(
            np.array(results).mean(axis=0), project.subjects)
