"""Evaluation metrics for Annif"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from sklearn.metrics import f1_score, precision_score, recall_score

from annif.exception import NotSupportedException
from annif.suggestion import SuggestionBatch, filter_suggestion

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from io import TextIOWrapper

    from click.utils import LazyFile
    from scipy.sparse._arrays import csr_array

    from annif.corpus.subject import SubjectIndex, SubjectSet
    from annif.suggestion import SubjectSuggestion


def true_positives(y_true: csr_array, y_pred: csr_array) -> int:
    """calculate the number of true positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return int((y_true.multiply(y_pred)).sum())


def false_positives(y_true: csr_array, y_pred: csr_array) -> int:
    """calculate the number of false positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return int((y_true < y_pred).sum())


def false_negatives(y_true: csr_array, y_pred: csr_array) -> int:
    """calculate the number of false negatives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return int((y_true > y_pred).sum())


def dcg_score(
    y_true: csr_array, y_pred: csr_array, limit: int | None = None
) -> np.float64:
    """return the discounted cumulative gain (DCG) score for the selected
    labels vs. relevant labels"""

    n_pred = y_pred.count_nonzero()
    if limit is not None:
        n_pred = min(limit, n_pred)

    top_k = y_pred.data.argsort()[-n_pred:][::-1]
    order = y_pred.indices[top_k]
    gain = y_true[:, order]
    discount = np.log2(np.arange(1, n_pred + 1) + 1)
    return (gain / discount).sum()


def ndcg_score(y_true: csr_array, y_pred: csr_array, limit: int | None = None) -> float:
    """return the normalized discounted cumulative gain (nDCG) score for the
    selected labels vs. relevant labels"""

    scores = np.ones(y_true.shape[0], dtype=np.float32)
    for i in range(y_true.shape[0]):
        true = y_true.getrow(i)
        idcg = dcg_score(true, true, limit)
        if idcg > 0:
            pred = y_pred.getrow(i)
            dcg = dcg_score(true, pred, limit)
            scores[i] = dcg / idcg

    return float(scores.mean())


class EvaluationBatch:
    """A class for evaluating batches of results using all available metrics.
    The evaluate() method is called once per document in the batch or evaluate_many()
    for a list of documents of the batch. Final results can be queried using the
    results() method."""

    def __init__(self, subject_index: SubjectIndex) -> None:
        self._subject_index = subject_index
        self._suggestion_arrays = []
        self._gold_subject_arrays = []

    def evaluate_many(
        self,
        suggestion_batch: list[list[SubjectSuggestion]]
        | SuggestionBatch
        | list[Iterator],
        gold_subject_batch: Sequence[SubjectSet],
    ) -> None:
        if not isinstance(suggestion_batch, SuggestionBatch):
            suggestion_batch = SuggestionBatch.from_sequence(
                suggestion_batch, self._subject_index
            )
        self._suggestion_arrays.append(suggestion_batch.array)

        # convert gold_subject_batch to sparse matrix
        ar = scipy.sparse.dok_array(
            (len(gold_subject_batch), len(self._subject_index)), dtype=bool
        )
        for idx, subject_set in enumerate(gold_subject_batch):
            for subject_id in subject_set:
                ar[idx, subject_id] = True
        self._gold_subject_arrays.append(ar.tocsr())

    def _evaluate_samples(
        self,
        y_true: csr_array,
        y_pred: csr_array,
        metrics: Iterable[str] = [],
    ) -> dict[str, float]:
        y_pred_binary = y_pred > 0.0

        # define the available metrics as lazy lambda functions
        # so we can execute only the ones actually requested
        all_metrics = {
            "Precision (doc avg)": lambda: precision_score(
                y_true, y_pred_binary, average="samples"
            ),
            "Recall (doc avg)": lambda: recall_score(
                y_true, y_pred_binary, average="samples"
            ),
            "F1 score (doc avg)": lambda: f1_score(
                y_true, y_pred_binary, average="samples"
            ),
            "Precision (subj avg)": lambda: precision_score(
                y_true, y_pred_binary, average="macro"
            ),
            "Recall (subj avg)": lambda: recall_score(
                y_true, y_pred_binary, average="macro"
            ),
            "F1 score (subj avg)": lambda: f1_score(
                y_true, y_pred_binary, average="macro"
            ),
            "Precision (weighted subj avg)": lambda: precision_score(
                y_true, y_pred_binary, average="weighted"
            ),
            "Recall (weighted subj avg)": lambda: recall_score(
                y_true, y_pred_binary, average="weighted"
            ),
            "F1 score (weighted subj avg)": lambda: f1_score(
                y_true, y_pred_binary, average="weighted"
            ),
            "Precision (microavg)": lambda: precision_score(
                y_true, y_pred_binary, average="micro"
            ),
            "Recall (microavg)": lambda: recall_score(
                y_true, y_pred_binary, average="micro"
            ),
            "F1 score (microavg)": lambda: f1_score(
                y_true, y_pred_binary, average="micro"
            ),
            "F1@5": lambda: f1_score(
                y_true, filter_suggestion(y_pred, 5) > 0.0, average="samples"
            ),
            "NDCG": lambda: ndcg_score(y_true, y_pred),
            "NDCG@5": lambda: ndcg_score(y_true, y_pred, limit=5),
            "NDCG@10": lambda: ndcg_score(y_true, y_pred, limit=10),
            "Precision@1": lambda: precision_score(
                y_true, filter_suggestion(y_pred, 1) > 0.0, average="samples"
            ),
            "Precision@3": lambda: precision_score(
                y_true, filter_suggestion(y_pred, 3) > 0.0, average="samples"
            ),
            "Precision@5": lambda: precision_score(
                y_true, filter_suggestion(y_pred, 5) > 0.0, average="samples"
            ),
            "True positives": lambda: true_positives(y_true, y_pred_binary),
            "False positives": lambda: false_positives(y_true, y_pred_binary),
            "False negatives": lambda: false_negatives(y_true, y_pred_binary),
        }

        if not metrics:
            metrics = all_metrics.keys()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return {metric: all_metrics[metric]() for metric in metrics}

    def _result_per_subject_header(
        self, results_file: LazyFile | TextIOWrapper
    ) -> None:
        print(
            "\t".join(
                [
                    "URI",
                    "Label",
                    "Support",
                    "True_positives",
                    "False_positives",
                    "False_negatives",
                    "Precision",
                    "Recall",
                    "F1_score",
                ]
            ),
            file=results_file,
        )

    def _result_per_subject_body(
        self, zipped_results: zip, results_file: LazyFile | TextIOWrapper
    ) -> None:
        for row in zipped_results:
            print("\t".join((str(e) for e in row)), file=results_file)

    def output_result_per_subject(
        self,
        y_true: csr_array,
        y_pred: csr_array,
        results_file: TextIOWrapper | LazyFile,
        language: str,
    ) -> None:
        """Write results per subject (non-aggregated)
        to outputfile results_file, using labels in the given language"""

        y_pred = y_pred.T > 0.0
        y_true = y_true.T

        true_pos = y_true.multiply(y_pred).sum(axis=1)
        false_pos = (y_true < y_pred).sum(axis=1)
        false_neg = (y_true > y_pred).sum(axis=1)

        with np.errstate(invalid="ignore"):
            precision = np.nan_to_num(true_pos / (true_pos + false_pos))
            recall = np.nan_to_num(true_pos / (true_pos + false_neg))
            f1_score = np.nan_to_num(2 * (precision * recall) / (precision + recall))

        zipped = zip(
            [subj.uri for subj in self._subject_index],  # URI
            [subj.labels[language] for subj in self._subject_index],  # Label
            y_true.sum(axis=1),  # Support
            true_pos,  # True positives
            false_pos,  # False positives
            false_neg,  # False negatives
            precision,  # Precision
            recall,  # Recall
            f1_score,  # F1 score
        )
        self._result_per_subject_header(results_file)
        self._result_per_subject_body(zipped, results_file)

    def results(
        self,
        metrics: Iterable[str] = [],
        results_file: LazyFile | TextIOWrapper | None = None,
        language: str | None = None,
    ) -> dict[str, float]:
        """evaluate a set of selected subjects against a gold standard using
        different metrics. If metrics is empty, use all available metrics.
        If results_file (file object) given, write results per subject to it
        with labels expressed in the given language."""

        if not self._suggestion_arrays:
            raise NotSupportedException("cannot evaluate empty corpus")

        y_pred = scipy.sparse.csr_array(scipy.sparse.vstack(self._suggestion_arrays))
        y_true = scipy.sparse.csr_array(scipy.sparse.vstack(self._gold_subject_arrays))

        results = self._evaluate_samples(y_true, y_pred, metrics)
        results["Documents evaluated"] = int(y_true.shape[0])

        if results_file:
            self.output_result_per_subject(y_true, y_pred, results_file, language)
        return results
