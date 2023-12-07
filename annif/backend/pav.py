"""PAV ensemble backend that combines results from multiple projects and
learns which concept suggestions from each backend are trustworthy using the
PAV algorithm, a.k.a. isotonic regression, to turn raw scores returned by
individual backends into probabilities."""
from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.isotonic import IsotonicRegression

import annif.corpus
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import ensemble

if TYPE_CHECKING:
    from annif.corpus.document import DocumentCorpus
    from annif.project import AnnifProject


class PAVBackend(ensemble.BaseEnsembleBackend):
    """PAV ensemble backend that combines results from multiple projects"""

    name = "pav"

    MODEL_FILE_PREFIX = "pav-model-"

    # defaults for uninitialized instances
    _models = None

    DEFAULT_PARAMETERS = {"min-docs": 10}

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        if self._models is not None:
            return  # already initialized
        self._models = {}
        sources = annif.util.parse_sources(self.params["sources"])
        for source_project_id, _ in sources:
            model_filename = self.MODEL_FILE_PREFIX + source_project_id
            path = os.path.join(self.datadir, model_filename)
            if os.path.exists(path):
                self.debug("loading PAV model from {}".format(path))
                self._models[source_project_id] = joblib.load(path)
            else:
                raise NotInitializedException(
                    "PAV model file '{}' not found".format(path),
                    backend_id=self.backend_id,
                )

    def _get_model(self, source_project_id: str) -> dict[int, IsotonicRegression]:
        self.initialize()
        return self._models[source_project_id]

    def _merge_source_batches(
        self,
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        reg_batch_by_source = {}
        for project_id, batch in batch_by_source.items():
            reg_models = self._get_model(project_id)
            pav_batch = [
                [
                    SubjectSuggestion(
                        subject_id=sugg.subject_id,
                        score=reg_models[sugg.subject_id].predict([sugg.score])[0],
                    )
                    if sugg.subject_id in reg_models
                    else SubjectSuggestion(
                        subject_id=sugg.subject_id, score=sugg.score
                    )  # default to raw score
                    for sugg in result
                ]
                for result in batch
            ]
            reg_batch_by_source[project_id] = SuggestionBatch.from_sequence(
                pav_batch, self.project.subjects
            )

        return super()._merge_source_batches(reg_batch_by_source, sources, params)

    @staticmethod
    def _suggest_train_corpus(
        source_project: AnnifProject, corpus: DocumentCorpus
    ) -> tuple[csc_matrix, csc_matrix]:
        # lists for constructing score matrix
        data, row, col = [], [], []
        # lists for constructing true label matrix
        trow, tcol = [], []

        ndocs = 0
        for docid, doc in enumerate(corpus.documents):
            hits = source_project.suggest([doc.text])[0]
            vector = hits.as_vector()
            for cid in np.flatnonzero(vector):
                data.append(vector[cid])
                row.append(docid)
                col.append(cid)
            for cid in np.flatnonzero(
                doc.subject_set.as_vector(len(source_project.subjects))
            ):
                trow.append(docid)
                tcol.append(cid)
            ndocs += 1
        scores = coo_matrix(
            (data, (row, col)),
            shape=(ndocs, len(source_project.subjects)),
            dtype=np.float32,
        )
        true = coo_matrix(
            (np.ones(len(trow), dtype=bool), (trow, tcol)),
            shape=(ndocs, len(source_project.subjects)),
            dtype=bool,
        )
        return csc_matrix(scores), csc_matrix(true)

    def _create_pav_model(
        self, source_project_id: str, min_docs: int, corpus: DocumentCorpus
    ) -> None:
        self.info(
            "creating PAV model for source {}, min_docs={}".format(
                source_project_id, min_docs
            )
        )
        source_project = self.project.registry.get_project(source_project_id)
        # suggest subjects for the training corpus
        scores, true = self._suggest_train_corpus(source_project, corpus)
        # create the concept-specific PAV regression models
        pav_regressions = {}
        for cid in range(len(source_project.subjects)):
            if true[:, cid].sum() < min_docs:
                continue  # don't create model b/c of too few examples
            reg = IsotonicRegression(out_of_bounds="clip")
            cid_scores = scores[:, cid].toarray().flatten().astype(np.float64)
            reg.fit(cid_scores, true[:, cid].toarray().flatten())
            pav_regressions[cid] = reg
        self.info("created PAV model for {} concepts".format(len(pav_regressions)))
        model_filename = self.MODEL_FILE_PREFIX + source_project_id
        annif.util.atomic_save(
            pav_regressions, self.datadir, model_filename, method=joblib.dump
        )

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus == "cached":
            raise NotSupportedException(
                "Training pav project from cached data not supported."
            )
        if corpus.is_empty():
            raise NotSupportedException(
                "training backend {} with no documents".format(self.backend_id)
            )
        self.info("creating PAV models")
        sources = annif.util.parse_sources(self.params["sources"])
        min_docs = int(params["min-docs"])
        for source_project_id, _ in sources:
            self._create_pav_model(source_project_id, min_docs, corpus)
