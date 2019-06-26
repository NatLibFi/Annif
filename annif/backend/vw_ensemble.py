"""Annif backend using the Vowpal Wabbit multiclass and multilabel
classifiers"""

import random
import os.path
import annif.util
from vowpalwabbit import pyvw
import numpy as np
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

    def _merge_hits_from_sources(self, hits_from_sources, project, params):
        score_vector = np.array([hits.vector
                                 for hits, _ in hits_from_sources])
        result = np.zeros(score_vector.shape[1])
        for subj_id in range(score_vector.shape[1]):
            if score_vector[:, subj_id].sum() > 0.0:
                ex = self._format_example(
                    subj_id,
                    score_vector[:, subj_id])
                score = (self._model.predict(ex) + 1.0) / 2.0
                result[subj_id] = score
        return VectorSuggestionResult(result, project.subjects)

    def _format_example(self, subject_id, scores, true=None):
        if true is None:
            val = ''
        elif true:
            val = 1
        else:
            val = -1
        ex = "{} |{}".format(val, subject_id)
        for proj_idx, proj in enumerate(self.source_project_ids):
            ex += " {}:{}".format(proj, scores[proj_idx])
        return ex

    @property
    def source_project_ids(self):
        sources = annif.util.parse_sources(self.params['sources'])
        return [project_id for project_id, _ in sources]

    def _create_examples(self, corpus, project):
        source_projects = [annif.project.get_project(project_id)
                           for project_id in self.source_project_ids]
        examples = []
        for doc in corpus.documents:
            subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
            true = subjects.as_vector(project.subjects)
            score_vectors = []
            for source_project in source_projects:
                hits = source_project.suggest(doc.text)
                score_vectors.append(hits.vector)
            score_vector = np.array(score_vectors)
            for subj_id in range(len(true)):
                if true[subj_id] or score_vector[:, subj_id].sum() > 0.0:
                    ex = self._format_example(
                        subj_id,
                        score_vector[:, subj_id],
                        true[subj_id])
                    examples.append(ex)
        random.shuffle(examples)
        return examples

    def _create_model(self, project):
        trainpath = os.path.join(self.datadir, self.TRAIN_FILE)
        params = self._create_params(
            {'data': trainpath, 'q': '::'})
        if params.get('passes', 1) > 1:
            # need a cache file when there are multiple passes
            params.update({'cache': True, 'kill_cache': True})
        self.debug("model parameters: {}".format(params))
        self._model = pyvw.vw(**params)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)
