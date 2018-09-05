"""Evaluation metrics for Annif"""

import collections
import statistics
import numpy


def precision(selected, relevant):
    """return the precision, i.e. the fraction of selected instances that
    are relevant"""
    sel = set(selected)
    rel = set(relevant)
    if len(sel) == 0:
        return 0.0  # avoid division by zero
    return len(sel & rel) / len(sel)


def precision_1(selected, relevant):
    return precision(selected[:1], relevant)


def precision_3(selected, relevant):
    return precision(selected[:3], relevant)


def precision_5(selected, relevant):
    return precision(selected[:5], relevant)


def recall(selected, relevant):
    """return the recall, i.e. the fraction of relevant instances that were
    selected"""
    sel = set(selected)
    rel = set(relevant)
    if len(rel) == 0:
        return 0.0  # avoid division by zero
    return len(sel & rel) / len(rel)


def f_measure(A, B):
    """return the F-measure similarity of two sets"""
    setA = set(A)
    setB = set(B)
    if len(setA) == 0 or len(setB) == 0:
        return 0.0  # shortcut, avoid division by zero
    return 2.0 * len(setA & setB) / (len(setA) + len(setB))


def true_positives(selected, relevant):
    """return the number of true positives, i.e. how many selected instances
    were relevant"""
    sel = set(selected)
    rel = set(relevant)
    return len(sel & rel)


def false_positives(selected, relevant):
    """return the number of false positives, i.e. how many selected instances
    were not relevant"""
    sel = set(selected)
    rel = set(relevant)
    return len(sel - rel)


def false_negatives(selected, relevant):
    """return the number of false negaives, i.e. how many relevant instances
    were not selected"""
    sel = set(selected)
    rel = set(relevant)
    return len(rel - sel)


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
    dcg_val = dcg(selected, relevant, at_k)
    dcg_max = dcg(relevant, relevant, at_k)
    if dcg_max == 0.0:
        return 0.0
    return dcg_val / dcg_max


def normalized_dcg_5(selected, relevant):
    return normalized_dcg(selected, relevant, 5)


def normalized_dcg_10(selected, relevant):
    return normalized_dcg(selected, relevant, 10)


def evaluate(samples):
    """evaluate a set of selected subject against a gold standard using
    different metrics"""

    metrics = [
        ('Precision', precision, statistics.mean),
        ('Recall', recall, statistics.mean),
        ('F-measure', f_measure, statistics.mean),
        ('NDCG@5', normalized_dcg_5, statistics.mean),
        ('NDCG@10', normalized_dcg_10, statistics.mean),
        ('Precision@1', precision_1, statistics.mean),
        ('Precision@3', precision_3, statistics.mean),
        ('Precision@5', precision_5, statistics.mean),
        ('True positives', true_positives, sum),
        ('False positives', false_positives, sum),
        ('False negatives', false_negatives, sum)
    ]

    results = collections.OrderedDict()
    for metric_name, metric_fn, merge_fn in metrics:
        scores = [metric_fn(selected, gold)
                  for selected, gold in samples]
        results[metric_name] = merge_fn(scores)
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
