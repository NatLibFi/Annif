"""Evaluation metrics for Annif"""

import collections
import statistics
import warnings
import numpy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score


def sklearn_metric_score(selected, relevant, metric_fn):
    """call a sklearn metric function, converting the selected and relevant
       subjects into the multilabel indicator arrays expected by sklearn"""
    mlb = MultiLabelBinarizer()
    mlb.fit(list(relevant) + list(selected))
    y_true = mlb.transform(relevant)
    y_pred = mlb.transform(selected)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metric_fn(y_true, y_pred, average='samples')


def precision(selected, relevant, limit=None):
    """return the precision, i.e. the fraction of selected instances that
    are relevant"""
    if limit is not None:
        selected = [subjs[:limit] for subjs in selected]
    return sklearn_metric_score(selected, relevant, precision_score)


def recall(selected, relevant):
    """return the recall, i.e. the fraction of relevant instances that were
    selected"""
    return sklearn_metric_score(selected, relevant, recall_score)


def f_measure(selected, relevant):
    """return the F-measure similarity of two sets"""
    return sklearn_metric_score(selected, relevant, f1_score)


def true_positives_bitwise(y_true, y_pred, average):
    """calculate the number of true positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return (y_true & y_pred).sum()


def true_positives(selected, relevant):
    """return the number of true positives, i.e. how many selected instances
    were relevant"""
    return sklearn_metric_score(selected, relevant, true_positives_bitwise)


def false_positives_bitwise(y_true, y_pred, average):
    """calculate the number of false positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return (~y_true & y_pred).sum()


def false_positives(selected, relevant):
    """return the number of false positives, i.e. how many selected instances
    were not relevant"""
    return sklearn_metric_score(selected, relevant, false_positives_bitwise)


def false_negatives_bitwise(y_true, y_pred, average):
    """calculate the number of false negatives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return (y_true & ~y_pred).sum()


def false_negatives(selected, relevant):
    """return the number of false negaives, i.e. how many relevant instances
    were not selected"""
    return sklearn_metric_score(selected, relevant, false_negatives_bitwise)


def dcg(selected, relevant, limit):
    """return the discounted cumulative gain (DCG) score for the selected
    instances vs. relevant instances"""
    if len(selected) == 0 or len(relevant) == 0:
        return 0.0
    scores = numpy.array([int(item in relevant)
                          for item in list(selected)[:limit]])
    weights = numpy.log2(numpy.arange(2, scores.size + 2))
    return numpy.sum(scores / weights)


def normalized_dcg(selected, relevant, limit):
    """return the normalized discounted cumulative gain (nDCG) score for the
    selected instances vs. relevant instances"""

    scores = []
    for ssubj, rsubj in zip(selected, relevant):
        dcg_val = dcg(ssubj, rsubj, limit)
        dcg_max = dcg(rsubj, rsubj, limit)
        if dcg_max == 0.0:
            scores.append(0.0)
        else:
            scores.append(dcg_val / dcg_max)
    return statistics.mean(scores)


class EvaluationBatch:
    """A class for evaluating batches of results using all available metrics.
    The evaluate() method is called once per document in the batch.
    Final results can be queried using the results() method."""

    def __init__(self):
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
        hits, gold_subjects = zip(*transformed_samples)

        results = collections.OrderedDict([
            ('Precision', precision(hits, gold_subjects)),
            ('Recall', recall(hits, gold_subjects)),
            ('F-measure', f_measure(hits, gold_subjects)),
            ('NDCG@5', normalized_dcg(hits, gold_subjects, limit=5)),
            ('NDCG@10', normalized_dcg(hits, gold_subjects, limit=10)),
            ('Precision@1', precision(hits, gold_subjects, limit=1)),
            ('Precision@3', precision(hits, gold_subjects, limit=3)),
            ('Precision@5', precision(hits, gold_subjects, limit=5)),
            ('True positives', true_positives(hits, gold_subjects)),
            ('False positives', false_positives(hits, gold_subjects)),
            ('False negatives', false_negatives(hits, gold_subjects))
        ])

        return results
