"""Unit tests for evaluation metrics in Annif"""

import numpy as np
import annif.corpus
import annif.eval
import annif.hit


def test_precision_at_k():
    y_true = np.array([[1, 0, 1, 0, 1, 0]])
    y_pred = np.array([[6, 5, 4, 3, 2, 1]])
    prec_10 = annif.eval.precision_at_k_score(y_true, y_pred, 10)
    assert prec_10 == 0.5
    prec_5 = annif.eval.precision_at_k_score(y_true, y_pred, 5)
    assert prec_5 == 0.6
    prec_4 = annif.eval.precision_at_k_score(y_true, y_pred, 4)
    assert prec_4 == 0.5
    prec_3 = annif.eval.precision_at_k_score(y_true, y_pred, 3)
    assert prec_3 > 0.66
    assert prec_3 < 0.67
    prec_1 = annif.eval.precision_at_k_score(y_true, y_pred, 1)
    assert prec_1 == 1.0


# DCG@6 example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
def test_dcg():
    y_true = np.array([3, 2, 3, 0, 1, 2])
    y_pred = np.array([6, 5, 4, 3, 2, 1])
    dcg = annif.eval.dcg_score(y_true, y_pred, 6)
    assert dcg > 6.86
    assert dcg < 6.87


# iDCG@6 example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
def test_dcg_ideal():
    y_true = np.array([3, 3, 3, 2, 2, 2, 1, 0])
    y_pred = np.array([8, 7, 6, 5, 4, 3, 2, 1])
    dcg = annif.eval.dcg_score(y_true, y_pred, 6)
    assert dcg > 8.74
    assert dcg < 8.75


# nDCG@6 example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
def test_ndcg():
    y_true = np.array([[3, 2, 3, 0, 1, 2, 3, 2]])
    y_pred = np.array([[6, 5, 4, 3, 2, 1, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred, 6)
    assert ndcg > 0.78
    assert ndcg < 0.79


def test_ndcg_nolimit():
    y_true = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    y_pred = np.array([[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred)
    assert ndcg > 0.49
    assert ndcg < 0.50


def test_ndcg_10():
    y_true = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    y_pred = np.array([[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred, 10)
    assert ndcg > 0.55
    assert ndcg < 0.56


def test_ndcg_5():
    y_true = np.array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    y_pred = np.array([[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred, 5)
    assert ndcg > 0.85
    assert ndcg < 0.86


def test_ndcg_empty():
    y_true = np.array([[1, 1, 1, 1, 1]])
    y_pred = np.array([[0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred)
    assert ndcg == 0


def test_ndcg_empty2():
    y_true = np.array([[0, 0, 0, 0, 0]])
    y_pred = np.array([[1, 1, 1, 1, 1]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred)
    assert ndcg == 1.0


def test_evaluation_batch(subject_index):
    batch = annif.eval.EvaluationBatch(subject_index)

    gold_set = annif.corpus.SubjectSet(
        '<http://www.yso.fi/onto/yso/p10849>\tarkeologit')
    hits1 = annif.hit.ListAnalysisResult([
        annif.hit.AnalysisHit(
            uri='http://www.yso.fi/onto/yso/p10849',
            label='arkeologit',
            score=1.0)], subject_index)
    batch.evaluate(hits1, gold_set)
    hits2 = annif.hit.ListAnalysisResult([
        annif.hit.AnalysisHit(
            uri='http://www.yso.fi/onto/yso/p1747',
            label='egyptologit',
            score=1.0)], subject_index)
    batch.evaluate(hits2, gold_set)
    results = batch.results()
    assert results['Precision (doc avg)'] == 0.5
    assert results['Recall (doc avg)'] == 0.5
    assert results['LRAP'] >= 0.50
    assert results['LRAP'] <= 0.51
    assert results['True positives'] == 1
    assert results['False positives'] == 1
    assert results['False negatives'] == 1
    assert results['Documents evaluated'] == 2
