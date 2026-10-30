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
        reg_batches = []
        for project_id, _ in sources:
            # align the source batch with this project's vocabulary, so that
            # the regression models below can be keyed by its subject IDs
            batch = self._align_batch(batch_by_source[project_id], project_id)
            reg_models = self._get_model(project_id)
            pav_batch = [
                [
                    (
                        SubjectSuggestion(
                            subject_id=sugg.subject_id,
                            score=reg_models[sugg.subject_id].predict([sugg.score])[0],
                        )
                        if sugg.subject_id in reg_models
                        else SubjectSuggestion(
                            subject_id=sugg.subject_id, score=sugg.score
                        )
                    )  # default to raw score
                    for sugg in result
                ]
                for result in batch
            ]
            reg_batches.append(
                SuggestionBatch.from_sequence(pav_batch, self.project.subjects)
            )

        # average the regressed batches directly; they are already aligned
        # with this project's vocabulary, so super()._merge_source_batches
        # must not try to align them again
        weights = [weight for _, weight in sources]
        return SuggestionBatch.from_averaged(reg_batches, weights).filter(
            limit=int(params["limit"])
        )

    def _suggest_train_corpus(
        self,
        source_project: AnnifProject,
        source_project_id: str,
        subject_map: np.ndarray | None,
        corpus: DocumentCorpus,
    ) -> tuple[csc_matrix, csc_matrix]:
        # lists for constructing score matrix
        data, row, col = [], [], []
        # lists for constructing true label matrix
        trow, tcol = [], []

        n_subjects = len(self.project.subjects)
        ndocs = 0
        for docid, doc in enumerate(corpus.documents):
            batch = source_project.suggest([doc])
            self._check_source_width(
                source_project_id, batch.array.shape[1], len(source_project.subjects)
            )
            hits = batch[0]
            # map the suggested source subject IDs onto ensemble subject IDs
            for suggestion in hits:
                subject_id = (
                    subject_map[suggestion.subject_id]
                    if subject_map is not None
                    else suggestion.subject_id
                )
                if subject_id >= 0:
                    data.append(suggestion.score)
                    row.append(docid)
                    col.append(subject_id)
            for subject_id in doc.subject_set:
                trow.append(docid)
                tcol.append(subject_id)
            ndocs += 1
        scores = coo_matrix(
            (data, (row, col)),
            shape=(ndocs, n_subjects),
            dtype=np.float32,
        )
        true = coo_matrix(
            (np.ones(len(trow), dtype=bool), (trow, tcol)),
            shape=(ndocs, n_subjects),
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
        # suggest subjects for the training corpus, aligned with this
        # project's vocabulary
        subject_map = self._source_subject_map(source_project_id)
        scores, true = self._suggest_train_corpus(
            source_project, source_project_id, subject_map, corpus
        )
        # create the concept-specific PAV regression models; only concepts
        # that the source can suggest need a model
        pav_regressions = {}
        if subject_map is None:
            concept_ids = range(len(self.project.subjects))
        else:
            concept_ids = np.unique(subject_map[subject_map >= 0])
        for cid in concept_ids:
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
