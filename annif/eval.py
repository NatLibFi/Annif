"""Evaluation metrics for Annif"""

import statistics
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import label_ranking_average_precision_score
from annif.exception import NotSupportedException


def filter_pred_top_k(preds, limit):
    """filter a 2D prediction vector, retaining only the top K suggestions
    for each individual prediction; the rest will be set to zeros"""

    masks = []
    for pred in preds:
        mask = np.zeros_like(pred, dtype=bool)
        top_k = np.argsort(pred)[::-1][:limit]
        mask[top_k] = True
        masks.append(mask)
    return preds * np.array(masks)


def true_positives(y_true, y_pred):
    """calculate the number of true positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return int((y_true & y_pred).sum())


def false_positives(y_true, y_pred):
    """calculate the number of false positives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return int((~y_true & y_pred).sum())


def false_negatives(y_true, y_pred):
    """calculate the number of false negatives using bitwise operations,
    emulating the way sklearn evaluation metric functions work"""
    return int((y_true & ~y_pred).sum())


def precision_at_k_score(y_true, y_pred, limit):
    """calculate the precision at K, i.e. the number of relevant items
    among the top K predicted ones"""
    scores = []
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

    def _evaluate_samples(self, y_true, y_pred, metrics=[]):
        y_pred_binary = y_pred > 0.0
        y_true_sparse = csr_matrix(y_true)

        # define the available metrics as lazy lambda functions
        # so we can execute only the ones actually requested
        all_metrics = {
            'Precision (doc avg)': lambda: precision_score(
                y_true_sparse, y_pred_binary, average='samples'),
            'Recall (doc avg)': lambda: recall_score(
                y_true_sparse, y_pred_binary, average='samples'),
            'F1 score (doc avg)': lambda: f1_score(
                y_true_sparse, y_pred_binary, average='samples'),
            'Precision (subj avg)': lambda: precision_score(
                y_true_sparse, y_pred_binary, average='macro'),
            'Recall (subj avg)': lambda: recall_score(
                y_true_sparse, y_pred_binary, average='macro'),
            'F1 score (subj avg)': lambda: f1_score(
                y_true_sparse, y_pred_binary, average='macro'),
            'Precision (weighted subj avg)': lambda: precision_score(
                y_true_sparse, y_pred_binary, average='weighted'),
            'Recall (weighted subj avg)': lambda: recall_score(
                y_true_sparse, y_pred_binary, average='weighted'),
            'F1 score (weighted subj avg)': lambda: f1_score(
                y_true_sparse, y_pred_binary, average='weighted'),
            'Precision (microavg)': lambda: precision_score(
                y_true_sparse, y_pred_binary, average='micro'),
            'Recall (microavg)': lambda: recall_score(
                y_true_sparse, y_pred_binary, average='micro'),
            'F1 score (microavg)': lambda: f1_score(
                y_true_sparse, y_pred_binary, average='micro'),
            'F1@5': lambda: f1_score(
                y_true_sparse,
                filter_pred_top_k(y_pred, 5) > 0.0,
                average='samples'),
            'NDCG': lambda: ndcg_score(y_true, y_pred),
            'NDCG@5': lambda: ndcg_score(y_true, y_pred, limit=5),
            'NDCG@10': lambda: ndcg_score(y_true, y_pred, limit=10),
            'Precision@1': lambda: precision_at_k_score(
                y_true, y_pred, limit=1),
            'Precision@3': lambda: precision_at_k_score(
                y_true, y_pred, limit=3),
            'Precision@5': lambda: precision_at_k_score(
                y_true, y_pred, limit=5),
            'LRAP': lambda: label_ranking_average_precision_score(
                y_true, y_pred),
            'True positives': lambda: true_positives(
                y_true, y_pred_binary),
            'False positives': lambda: false_positives(
                y_true, y_pred_binary),
            'False negatives': lambda: false_negatives(
                y_true, y_pred_binary),
        }

        if not metrics:
            metrics = all_metrics.keys()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            return {metric: all_metrics[metric]() for metric in metrics}

    def _result_per_subject_header(self, results_file):
        print('\t'.join(['URI',
                         'Label',
                         'Support',
                         'True_positives',
                         'False_positives',
                         'False_negatives',
                         'Precision',
                         'Recall',
                         'F1_score']),
              file=results_file)

    def _result_per_subject_body(self, zipped_results, results_file):
        for row in zipped_results:
            print('\t'.join((str(e) for e in row)), file=results_file)

    def output_result_per_subject(self, y_true, y_pred, results_file):
        """Write results per subject (non-aggregated)
        to outputfile results_file"""

        y_pred = y_pred.T > 0.0
        y_true = y_true.T > 0.0

        true_pos = (y_true & y_pred)
        false_pos = (~y_true & y_pred)
        false_neg = (y_true & ~y_pred)

        r = len(y_true)

        zipped = zip(self._subject_index._uris,               # URI
                     self._subject_index._labels,             # Label
                     np.sum((true_pos + false_neg), axis=1),  # Support
                     np.sum(true_pos, axis=1),                # True_positives
                     np.sum(false_pos, axis=1),               # False_positives
                     np.sum(false_neg, axis=1),               # False_negatives
                     [precision_score(y_true[i], y_pred[i], zero_division=0)
                      for i in range(r)],                     # Precision
                     [recall_score(y_true[i], y_pred[i], zero_division=0)
                      for i in range(r)],                     # Recall
                     [f1_score(y_true[i], y_pred[i], zero_division=0)
                      for i in range(r)])                     # F1
        self._result_per_subject_header(results_file)
        self._result_per_subject_body(zipped, results_file)

    def results(self, metrics=[], results_file=None, warnings=False):
        """evaluate a set of selected subjects against a gold standard using
        different metrics. If metrics is empty, use all available metrics.
        If results_file (file object) given, write results per subject to it"""

        if not self._samples:
            raise NotSupportedException("cannot evaluate empty corpus")

        shape = (len(self._samples), len(self._subject_index))
        y_true = np.zeros(shape, dtype=bool)
        y_pred = np.zeros(shape, dtype=np.float32)

        for idx, (hits, gold_subjects) in enumerate(self._samples):
            gold_subjects.as_vector(self._subject_index,
                                    destination=y_true[idx],
                                    warnings=warnings)
            hits.as_vector(self._subject_index, destination=y_pred[idx])

        results = self._evaluate_samples(y_true, y_pred, metrics)
        results['Documents evaluated'] = int(y_true.shape[0])

        if results_file:
            self.output_result_per_subject(y_true, y_pred, results_file)
        return results
