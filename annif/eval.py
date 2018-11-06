"""Evaluation metrics for Annif"""

import collections
import statistics
import warnings
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import label_ranking_average_precision_score


def true_positives(y_true, y_pred):
    """calculate the number of true positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return (y_true & y_pred).sum()


def false_positives(y_true, y_pred):
    """calculate the number of false positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return (~y_true & y_pred).sum()


def false_negatives(y_true, y_pred):
    """calculate the number of false negatives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return (y_true & ~y_pred).sum()


def precision_at_k_score(y_true, y_pred, limit):
    """calculate the precision at K, i.e. the number of relevant items
    among the top K predicted ones"""
    scores = []
    for true, pred in zip(y_true, y_pred):
        order = pred.argsort()[::-1]
        limit = min(limit, np.count_nonzero(pred))
        order = order[:limit]
        gain = true[order]
        scores.append(gain.sum() / limit)
    return statistics.mean(scores)


def dcg_score(y_true, y_pred, limit=None):
    """return the discounted cumulative gain (DCG) score for the selected
    labels vs. relevant labels"""
    order = y_pred.argsort()[::-1]
    n_pred = np.count_nonzero(y_pred)
    if limit is not None:
        n_pred = min(limit, n_pred)
    order = order[:n_pred]
    gain = y_true[order]
    discount = np.log2(np.arange(order.size) + 2)

    return (gain / discount).sum()


def ndcg_score(y_true, y_pred, limit=None):
    """return the normalized discounted cumulative gain (nDCG) score for the
    selected labels vs. relevant labels"""
    scores = []
    for true, pred in zip(y_true, y_pred):
        idcg = dcg_score(true, true, limit)
        dcg = dcg_score(true, pred, limit)
        if idcg > 0:
            scores.append(dcg / idcg)
        else:
            scores.append(1.0)  # perfect score for no relevant hits case
    return statistics.mean(scores)


class EvaluationBatch:
    """A class for evaluating batches of results using all available metrics.
    The evaluate() method is called once per document in the batch.
    Final results can be queried using the results() method."""

    def __init__(self, subject_index):
        self._subject_index = subject_index
        self._samples = []

    def evaluate(self, hits, gold_subjects):
        self._samples.append((hits, gold_subjects))

    def results(self):
        """evaluate a set of selected subjects against a gold standard using
        different metrics"""

        y_true = np.array([gold_subjects.as_vector(self._subject_index)
                           for hits, gold_subjects in self._samples])
        y_pred = np.array([hits.vector
                           for hits, gold_subjects in self._samples])
        y_pred_binary = y_pred > 0.0

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            results = collections.OrderedDict([
                ('Precision (doc avg)',
                 precision_score(y_true, y_pred_binary, average='samples')),
                ('Recall (doc avg)',
                 recall_score(y_true, y_pred_binary, average='samples')),
                ('F1 score (doc avg)',
                 f1_score(y_true, y_pred_binary, average='samples')),
                ('Precision (conc avg)',
                 precision_score(y_true, y_pred_binary, average='macro')),
                ('Recall (conc avg)',
                 recall_score(y_true, y_pred_binary, average='macro')),
                ('F1 score (conc avg)',
                 f1_score(y_true, y_pred_binary, average='macro')),
                ('Precision (microavg)',
                 precision_score(y_true, y_pred_binary, average='micro')),
                ('Recall (microavg)',
                 recall_score(y_true, y_pred_binary, average='micro')),
                ('F1 score (microavg)',
                 f1_score(y_true, y_pred_binary, average='micro')),
                ('NDCG', ndcg_score(y_true, y_pred)),
                ('NDCG@5', ndcg_score(y_true, y_pred, limit=5)),
                ('NDCG@10', ndcg_score(y_true, y_pred, limit=10)),
                ('Precision@1',
                 precision_at_k_score(y_true, y_pred, limit=1)),
                ('Precision@3',
                 precision_at_k_score(y_true, y_pred, limit=3)),
                ('Precision@5',
                 precision_at_k_score(y_true, y_pred, limit=5)),
                ('LRAP',
                 label_ranking_average_precision_score(y_true, y_pred)),
                ('True positives', true_positives(y_true, y_pred_binary)),
                ('False positives', false_positives(y_true, y_pred_binary)),
                ('False negatives', false_negatives(y_true, y_pred_binary)),
                ('Documents evaluated', y_true.shape[0])
            ])

        return results
