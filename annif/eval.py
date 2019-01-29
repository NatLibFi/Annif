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
    origlimit = limit
    for true, pred in zip(y_true, y_pred):
        order = pred.argsort()[::-1]
        orderlimit = min(limit, np.count_nonzero(pred))
        order = order[:orderlimit]
        gain = true[order]
        if orderlimit > 0:
            scores.append(gain.sum() / orderlimit)
        else:
            scores.append(0.0)
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

    def _evaluate_samples(self, y_true, y_pred, metrics='all'):
        y_pred_binary = y_pred > 0.0
        results = collections.OrderedDict()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            results['Precision (doc avg)'] = precision_score(
                y_true, y_pred_binary, average='samples')
            results['Recall (doc avg)'] = recall_score(
                y_true, y_pred_binary, average='samples')
            results['F1 score (doc avg)'] = f1_score(
                y_true, y_pred_binary, average='samples')
            if metrics == 'all':
                results['Precision (conc avg)'] = precision_score(
                    y_true, y_pred_binary, average='macro')
                results['Recall (conc avg)'] = recall_score(
                    y_true, y_pred_binary, average='macro')
                results['F1 score (conc avg)'] = f1_score(
                    y_true, y_pred_binary, average='macro')
                results['Precision (microavg)'] = precision_score(
                    y_true, y_pred_binary, average='micro')
                results['Recall (microavg)'] = recall_score(
                    y_true, y_pred_binary, average='micro')
                results['F1 score (microavg)'] = f1_score(
                    y_true, y_pred_binary, average='micro')
            results['NDCG'] = ndcg_score(y_true, y_pred)
            results['NDCG@5'] = ndcg_score(y_true, y_pred, limit=5)
            results['NDCG@10'] = ndcg_score(y_true, y_pred, limit=10)
            if metrics == 'all':
                results['Precision@1'] = precision_at_k_score(
                    y_true, y_pred, limit=1)
                results['Precision@3'] = precision_at_k_score(
                    y_true, y_pred, limit=3)
                results['Precision@5'] = precision_at_k_score(
                    y_true, y_pred, limit=5)
                results['LRAP'] = label_ranking_average_precision_score(
                    y_true, y_pred)
                results['True positives'] = true_positives(
                    y_true, y_pred_binary)
                results['False positives'] = false_positives(
                    y_true, y_pred_binary)
                results['False negatives'] = false_negatives(
                    y_true, y_pred_binary)

        return results

    def results(self, metrics='all'):
        """evaluate a set of selected subjects against a gold standard using
        different metrics. The set of metrics can be either 'all' or
        'simple'."""

        y_true = np.array([gold_subjects.as_vector(self._subject_index)
                           for hits, gold_subjects in self._samples])
        y_pred = np.array([hits.vector
                           for hits, gold_subjects in self._samples])

        results = self._evaluate_samples(
            y_true, y_pred, metrics)
        results['Documents evaluated'] = y_true.shape[0]
        return results
