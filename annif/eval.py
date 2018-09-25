"""Evaluation metrics for Annif"""

import collections
import statistics
import warnings
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
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

    def _transform_sample(self, sample):
        """transform a single document (sample) with predicted and gold
           standard subjects into either sequences of URIs (if available) or
           sequences of labels"""
        hits, gold_subjects = sample
        if gold_subjects.has_uris():
            selected = [hit.uri for hit in hits]
            gold_set = gold_subjects.subject_uris
        else:
            selected = [hit.label for hit in hits]
            gold_set = gold_subjects.subject_labels
        return (selected, gold_set)

    def results(self):
        """evaluate a set of selected subjects against a gold standard using
        different metrics"""

        transformed_samples = [self._transform_sample(sample)
                               for sample in self._samples]
        selected, relevant = zip(*transformed_samples)

        mlb = MultiLabelBinarizer()
        mlb.fit(list(selected) + list(relevant))
        y_true = mlb.transform(relevant)
        y_pred = mlb.transform(selected)

        y_true2 = np.array([gold_subjects.as_vector(self._subject_index)
                            for hits, gold_subjects in self._samples])

        y_scores = np.array([hits.as_vector(self._subject_index)
                             for hits, gold_subjects in self._samples])

        def y_pred_at(limit):
            return mlb.transform([subjs[:limit] for subjs in selected])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            results = collections.OrderedDict([
                ('Precision (per document average)',
                 precision_score(y_true, y_pred, average='samples')),
                ('Recall (per document average)',
                 recall_score(y_true, y_pred, average='samples')),
                ('F1 score (per document average)',
                 f1_score(y_true, y_pred, average='samples')),
                ('NDCG', ndcg_score(y_true2, y_scores)),
                ('NDCG@5', ndcg_score(y_true2, y_scores, limit=5)),
                ('NDCG@10', ndcg_score(y_true2, y_scores, limit=10)),
                ('Precision@1 (per document average)',
                 precision_score(y_true, y_pred_at(1), average='samples')),
                ('Precision@3 (per document average)',
                 precision_score(y_true, y_pred_at(3), average='samples')),
                ('Precision@5 (per document average)',
                 precision_score(y_true, y_pred_at(5), average='samples')),
                ('Label ranking average precision',
                 label_ranking_average_precision_score(y_true2, y_scores)),
                ('True positives', true_positives(y_true, y_pred)),
                ('False positives', false_positives(y_true, y_pred)),
                ('False negatives', false_negatives(y_true, y_pred))
            ])

        return results
