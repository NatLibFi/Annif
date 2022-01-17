"""PAV ensemble backend that combines results from multiple projects and
learns which concept suggestions from each backend are trustworthy using the
PAV algorithm, a.k.a. isotonic regression, to turn raw scores returned by
individual backends into probabilities."""

import os.path
import joblib
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.isotonic import IsotonicRegression
import numpy as np
import annif.corpus
import annif.suggestion
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from . import backend
from . import ensemble


class PAVBackend(ensemble.BaseEnsembleBackend):
    """PAV ensemble backend that combines results from multiple projects"""
    name = "pav"

    MODEL_FILE_PREFIX = "pav-model-"

    # defaults for uninitialized instances
    _models = None

    DEFAULT_PARAMETERS = {'min-docs': 10}

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def initialize(self, parallel=False):
        super().initialize(parallel)
        if self._models is not None:
            return  # already initialized
        self._models = {}
        sources = annif.util.parse_sources(self.params['sources'])
        for source_project_id, _ in sources:
            model_filename = self.MODEL_FILE_PREFIX + source_project_id
            path = os.path.join(self.datadir, model_filename)
            if os.path.exists(path):
                self.debug('loading PAV model from {}'.format(path))
                self._models[source_project_id] = joblib.load(path)
            else:
                raise NotInitializedException(
                    "PAV model file '{}' not found".format(path),
                    backend_id=self.backend_id)

    def _get_model(self, source_project_id):
        self.initialize()
        return self._models[source_project_id]

    def _normalize_hits(self, hits, source_project):
        reg_models = self._get_model(source_project.project_id)
        pav_result = []
        for hit in hits.as_list(source_project.subjects):
            if hit.uri in reg_models:
                score = reg_models[hit.uri].predict([hit.score])[0]
            else:  # default to raw score
                score = hit.score
            pav_result.append(
                annif.suggestion.SubjectSuggestion(
                    uri=hit.uri,
                    label=hit.label,
                    notation=hit.notation,
                    score=score))
        pav_result.sort(key=lambda hit: hit.score, reverse=True)
        return annif.suggestion.ListSuggestionResult(pav_result)

    @staticmethod
    def _suggest_train_corpus(source_project, corpus):
        # lists for constructing score matrix
        data, row, col = [], [], []
        # lists for constructing true label matrix
        trow, tcol = [], []

        ndocs = 0
        for docid, doc in enumerate(corpus.documents):
            hits = source_project.suggest(doc.text)
            vector = hits.as_vector(source_project.subjects)
            for cid in np.flatnonzero(vector):
                data.append(vector[cid])
                row.append(docid)
                col.append(cid)
            subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
            for cid in np.flatnonzero(
                    subjects.as_vector(source_project.subjects)):

                trow.append(docid)
                tcol.append(cid)
            ndocs += 1
        scores = coo_matrix((data, (row, col)),
                            shape=(ndocs, len(source_project.subjects)),
                            dtype=np.float32)
        true = coo_matrix((np.ones(len(trow), dtype=bool), (trow, tcol)),
                          shape=(ndocs, len(source_project.subjects)),
                          dtype=bool)
        return csc_matrix(scores), csc_matrix(true)

    def _create_pav_model(self, source_project_id, min_docs, corpus):
        self.info("creating PAV model for source {}, min_docs={}".format(
            source_project_id, min_docs))
        source_project = self.project.registry.get_project(source_project_id)
        # suggest subjects for the training corpus
        scores, true = self._suggest_train_corpus(source_project, corpus)
        # create the concept-specific PAV regression models
        pav_regressions = {}
        for cid in range(len(source_project.subjects)):
            if true[:, cid].sum() < min_docs:
                continue  # don't create model b/c of too few examples
            reg = IsotonicRegression(out_of_bounds='clip')
            cid_scores = scores[:, cid].toarray().flatten().astype(np.float64)
            reg.fit(cid_scores, true[:, cid].toarray().flatten())
            pav_regressions[source_project.subjects[cid][0]] = reg
        self.info("created PAV model for {} concepts".format(
            len(pav_regressions)))
        model_filename = self.MODEL_FILE_PREFIX + source_project_id
        annif.util.atomic_save(
            pav_regressions,
            self.datadir,
            model_filename,
            method=joblib.dump)

    def _train(self, corpus, params, jobs=0):
        if corpus == 'cached':
            raise NotSupportedException(
                'Training pav project from cached data not supported.')
        if corpus.is_empty():
            raise NotSupportedException('training backend {} with no documents'
                                        .format(self.backend_id))
        self.info("creating PAV models")
        sources = annif.util.parse_sources(self.params['sources'])
        min_docs = int(params['min-docs'])
        for source_project_id, _ in sources:
            self._create_pav_model(source_project_id, min_docs, corpus)
