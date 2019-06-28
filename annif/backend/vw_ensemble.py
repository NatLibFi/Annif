"""Annif backend using the Vowpal Wabbit multiclass and multilabel
classifiers"""

import collections
import json
import random
import os.path
import annif.util
import annif.project
import numpy as np
from annif.exception import NotInitializedException
from annif.suggestion import VectorSuggestionResult
from . import vw_base
from . import ensemble


class VWEnsembleBackend(
        ensemble.EnsembleBackend,
        vw_base.VWBaseBackend):
    """Vowpal Wabbit ensemble backend that combines results from multiple
    projects and learns how well those projects/backends recognize
    particular subjects."""

    name = "vw_ensemble"

    VW_PARAMS = {
        'bit_precision': (int, None),
        'learning_rate': (float, None),
        'loss_function': (['squared', 'logistic', 'hinge'], 'squared'),
        'l1': (float, None),
        'l2': (float, None),
        'passes': (int, None)
    }

    # number of training examples per subject, stored as a collections.Counter
    _subject_freq = None

    FREQ_FILE = 'subject-freq.json'

    # The discount rate affects how quickly the ensemble starts to trust its
    # own judgement when the amount of training data increases, versus using
    # a simple mean of scores. A higher value will mean that the model
    # adapts quicker (and possibly makes more errors) while a lower value
    # will make it more careful so that it will require more training data.
    DEFAULT_DISCOUNT_RATE = 0.01

    def _load_subject_freq(self):
        path = os.path.join(self.datadir, self.FREQ_FILE)
        if not os.path.exists(path):
            raise NotInitializedException(
                'frequency file {} not found'.format(path),
                backend_id=self.backend_id)
        self.debug('loading concept frequencies from {}'.format(path))
        with open(path) as freqf:
            # The Counter was serialized like a dictionary, need to
            # convert it back. Keys that became strings need to be turned
            # back into integers.
            self._subject_freq = collections.Counter()
            for cid, freq in json.load(freqf).items():
                self._subject_freq[int(cid)] = freq
        self.debug('loaded frequencies for {} concepts'.format(
            len(self._subject_freq)))

    def initialize(self):
        if self._subject_freq is None:
            self._load_subject_freq()
        super().initialize()

    def _calculate_scores(self, subj_id, subj_score_vector):
        ex = self._format_example(subj_id, subj_score_vector)
        raw_score = subj_score_vector.mean()
        pred_score = (self._model.predict(ex) + 1.0) / 2.0
        return raw_score, pred_score

    def _merge_hits_from_sources(self, hits_from_sources, project, params):
        score_vector = np.array([hits.vector
                                 for hits, _ in hits_from_sources])
        discount_rate = float(self.params.get('discount_rate',
                                              self.DEFAULT_DISCOUNT_RATE))
        result = np.zeros(score_vector.shape[1])
        for subj_id in range(score_vector.shape[1]):
            subj_score_vector = score_vector[:, subj_id]
            if subj_score_vector.sum() > 0.0:
                raw_score, pred_score = self._calculate_scores(
                    subj_id, subj_score_vector)
                raw_weight = 1.0 / \
                    ((discount_rate * self._subject_freq[subj_id]) + 1)
                result[subj_id] = (raw_weight * raw_score) + \
                    (1.0 - raw_weight) * pred_score
        return VectorSuggestionResult(result, project.subjects)

    @property
    def _source_project_ids(self):
        sources = annif.util.parse_sources(self.params['sources'])
        return [project_id for project_id, _ in sources]

    def _format_example(self, subject_id, scores, true=None):
        if true is None:
            val = ''
        elif true:
            val = 1
        else:
            val = -1
        ex = "{} |{}".format(val, subject_id)
        for proj_idx, proj in enumerate(self._source_project_ids):
            ex += " {}:{:.6f}".format(proj, scores[proj_idx])
        return ex

    def _doc_score_vector(self, doc, source_projects):
        score_vectors = []
        for source_project in source_projects:
            hits = source_project.suggest(doc.text)
            score_vectors.append(hits.vector)
        return np.array(score_vectors)

    def _doc_to_example(self, doc, project, source_projects):
        examples = []
        subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
        true = subjects.as_vector(project.subjects)
        score_vector = self._doc_score_vector(doc, source_projects)
        for subj_id in range(len(true)):
            if true[subj_id] or score_vector[:, subj_id].sum() > 0.0:
                ex = (subj_id, self._format_example(
                    subj_id,
                    score_vector[:, subj_id],
                    true[subj_id]))
                examples.append(ex)
        return examples

    def _create_examples(self, corpus, project):
        source_projects = [annif.project.get_project(project_id)
                           for project_id in self._source_project_ids]
        examples = []
        for doc in corpus.documents:
            examples += self._doc_to_example(doc, project, source_projects)
        random.shuffle(examples)
        return examples

    @staticmethod
    def _write_freq_file(subject_freq, filename):
        with open(filename, 'w') as freqfile:
            json.dump(subject_freq, freqfile)

    def _create_train_file(self, corpus, project):
        self.info('creating VW train file')
        exampledata = self._create_examples(corpus, project)

        subjects = [subj_id for subj_id, ex in exampledata]
        self._subject_freq = collections.Counter(subjects)
        annif.util.atomic_save(self._subject_freq,
                               self.datadir,
                               self.FREQ_FILE,
                               method=self._write_freq_file)

        examples = [ex for subj_id, ex in exampledata]
        annif.util.atomic_save(examples,
                               self.datadir,
                               self.TRAIN_FILE,
                               method=self._write_train_file)

    def learn(self, corpus, project):
        self.initialize()
        exampledata = self._create_examples(corpus, project)
        for subj_id, example in exampledata:
            self._model.learn(example)
            self._subject_freq[subj_id] += 1
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)
        annif.util.atomic_save(self._subject_freq,
                               self.datadir,
                               self.FREQ_FILE,
                               method=self._write_freq_file)
