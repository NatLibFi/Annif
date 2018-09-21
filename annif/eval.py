"""Evaluation metrics for Annif"""

import collections
import statistics
import warnings
import numpy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score


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
        selected, relevant = zip(*transformed_samples)

        mlb = MultiLabelBinarizer()
        mlb.fit(list(selected) + list(relevant))
        y_true = mlb.transform(relevant)
        y_pred = mlb.transform(selected)

        def y_pred_at(limit):
            return mlb.transform([subjs[:limit] for subjs in selected])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            results = collections.OrderedDict([
                ('Precision', precision_score(y_true,
                                              y_pred,
                                              average='samples')),
                ('Recall', recall_score(y_true, y_pred, average='samples')),
                ('F-measure', f1_score(y_true, y_pred, average='samples')),
                ('NDCG@5', normalized_dcg(selected, relevant, limit=5)),
                ('NDCG@10', normalized_dcg(selected, relevant, limit=10)),
                ('Precision@1', precision_score(y_true,
                                                y_pred_at(1),
                                                average='samples')),
                ('Precision@3', precision_score(y_true,
                                                y_pred_at(3),
                                                average='samples')),
                ('Precision@5', precision_score(y_true,
                                                y_pred_at(5),
                                                average='samples')),
                ('True positives', true_positives(y_true, y_pred)),
                ('False positives', false_positives(y_true, y_pred)),
                ('False negatives', false_negatives(y_true, y_pred))
            ])

        return results
