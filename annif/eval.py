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


def evaluate(selected, gold):
    """evaluate a set of selected subject against a gold standard using
    different metrics"""
    return [
        ('Precision', precision(selected, gold), statistics.mean),
        ('Recall', recall(selected, gold), statistics.mean),
        ('F-measure', f_measure(selected, gold), statistics.mean),
        ('NDCG@5', normalized_dcg(selected, gold, 5), statistics.mean),
        ('NDCG@10', normalized_dcg(selected, gold, 10), statistics.mean),
        ('Precision@1', precision(selected[:1], gold), statistics.mean),
        ('Precision@3', precision(selected[:3], gold), statistics.mean),
        ('Precision@5', precision(selected[:5], gold), statistics.mean),
        ('True positives', true_positives(selected, gold), sum),
        ('False positives', false_positives(selected, gold), sum),
        ('False negatives', false_negatives(selected, gold), sum)
    ]


def evaluate_hits(hits, gold_subjects):
    """evaluate a list of AnalysisHit objects against a SubjectSet,
    returning evaluation metrics"""
    if gold_subjects.has_uris():
        selected = [hit.uri for hit in hits]
        gold_set = gold_subjects.subject_uris
    else:
        selected = [hit.label for hit in hits]
        gold_set = gold_subjects.subject_labels

    return evaluate(selected, gold_set)


class EvaluationBatch:
    """A class for evaluating batches of results using all available metrics.
    The evaluate() method is called once per document in the batch.
    Final results can be queried using the results() method."""

    def __init__(self):
        self._results = []

    def evaluate(self, hits, gold_subjects):
        self._results.append(evaluate_hits(hits, gold_subjects))

    def results(self):
        measures = collections.OrderedDict()
        merge_functions = {}
        for result in self._results:
            for metric, score, merge_function in result:
                measures.setdefault(metric, [])
                measures[metric].append(score)
                merge_functions[metric] = merge_function
        final_results = collections.OrderedDict()
        for metric, results in measures.items():
            score = merge_functions[metric](results)
            final_results[metric] = score
        return final_results
