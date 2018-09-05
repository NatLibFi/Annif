"""Evaluation metrics for Annif"""

import collections
import statistics
import numpy
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score


def precision(selected, relevant):
    """return the precision, i.e. the fraction of selected instances that
    are relevant"""
    mlb = MultiLabelBinarizer()
    mlb.fit(list(relevant) + list(selected))
    y_true = mlb.transform(relevant)
    y_pred = mlb.transform(selected)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return precision_score(y_true, y_pred, average='samples')


def precision_1(selected, relevant):
    return precision([subjs[:1] for subjs in selected], relevant)


def precision_3(selected, relevant):
    return precision([subjs[:3] for subjs in selected], relevant)


def precision_5(selected, relevant):
    return precision([subjs[:5] for subjs in selected], relevant)


def recall(selected, relevant):
    """return the recall, i.e. the fraction of relevant instances that were
    selected"""
    scores = []
    for ssubj, rsubj in zip(selected, relevant):
        sel = set(ssubj)
        rel = set(rsubj)
        if len(rel) == 0:
            scores.append(0.0)  # avoid division by zero
        else:
            scores.append(len(sel & rel) / len(rel))
    return statistics.mean(scores)


def f_measure(A, B):
    """return the F-measure similarity of two sets"""
    scores = []
    for asubj, bsubj in zip(A, B):
        setA = set(asubj)
        setB = set(bsubj)
        if len(setA) == 0 or len(setB) == 0:
            scores.append(0.0)  # shortcut, avoid division by zero
        else:
            scores.append(2.0 * len(setA & setB) / (len(setA) + len(setB)))
    return statistics.mean(scores)


def true_positives(selected, relevant):
    """return the number of true positives, i.e. how many selected instances
    were relevant"""
    count = 0
    for ssubj, rsubj in zip(selected, relevant):
        sel = set(ssubj)
        rel = set(rsubj)
        count += len(sel & rel)
    return count


def false_positives(selected, relevant):
    """return the number of false positives, i.e. how many selected instances
    were not relevant"""
    count = 0
    for ssubj, rsubj in zip(selected, relevant):
        sel = set(ssubj)
        rel = set(rsubj)
        count += len(sel - rel)
    return count


def false_negatives(selected, relevant):
    """return the number of false negaives, i.e. how many relevant instances
    were not selected"""
    count = 0
    for ssubj, rsubj in zip(selected, relevant):
        sel = set(ssubj)
        rel = set(rsubj)
        count += len(rel - sel)
    return count


def dcg(selected, relevant, at_k):
    """return the discounted cumulative gain (DCG) score for the selected
    instances vs. relevant instances"""
    if len(selected) == 0 or len(relevant) == 0:
        return 0.0
    scores = numpy.array([int(item in relevant)
                          for item in list(selected)[:at_k]])
    weights = numpy.log2(numpy.arange(2, scores.size + 2))
    return numpy.sum(scores / weights)


def normalized_dcg(selected, relevant, at_k):
    """return the normalized discounted cumulative gain (nDCG) score for the
    selected instances vs. relevant instances"""

    scores = []
    for ssubj, rsubj in zip(selected, relevant):
        dcg_val = dcg(ssubj, rsubj, at_k)
        dcg_max = dcg(rsubj, rsubj, at_k)
        if dcg_max == 0.0:
            scores.append(0.0)
        else:
            scores.append(dcg_val / dcg_max)
    return statistics.mean(scores)


def normalized_dcg_5(selected, relevant):
    return normalized_dcg(selected, relevant, 5)


def normalized_dcg_10(selected, relevant):
    return normalized_dcg(selected, relevant, 10)


def evaluate(samples):
    """evaluate a set of selected subject against a gold standard using
    different metrics"""

    metrics = [
        ('Precision', precision),
        ('Recall', recall),
        ('F-measure', f_measure),
        ('NDCG@5', normalized_dcg_5),
        ('NDCG@10', normalized_dcg_10),
        ('Precision@1', precision_1),
        ('Precision@3', precision_3),
        ('Precision@5', precision_5),
        ('True positives', true_positives),
        ('False positives', false_positives),
        ('False negatives', false_negatives)
    ]

    results = collections.OrderedDict()
    for metric_name, metric_fn in metrics:
        hits, gold_subjects = zip(*samples)
        results[metric_name] = metric_fn(hits, gold_subjects)
    return results


def transform_sample(sample):
    hits, gold_subjects = sample
    if gold_subjects.has_uris():
        selected = [hit.uri for hit in hits]
        gold_set = gold_subjects.subject_uris
    else:
        selected = [hit.label for hit in hits]
        gold_set = gold_subjects.subject_labels
    return (selected, gold_set)


class EvaluationBatch:
    """A class for evaluating batches of results using all available metrics.
    The evaluate() method is called once per document in the batch.
    Final results can be queried using the results() method."""

    def __init__(self):
        self._samples = []

    def evaluate(self, hits, gold_subjects):
        self._samples.append((hits, gold_subjects))

    def results(self):
        transformed_samples = [transform_sample(sample)
                               for sample in self._samples]
        return evaluate(transformed_samples)
