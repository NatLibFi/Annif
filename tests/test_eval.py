"""Unit tests for evaluation metrics in Annif"""

import numpy as np
import annif.corpus
import annif.eval
import annif.hit


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
    hits1 = annif.hit.AnalysisResult([
        annif.hit.AnalysisHit(
            uri='http://www.yso.fi/onto/yso/p10849',
            label='arkeologit',
            score=1.0)])
    batch.evaluate(hits1, gold_set)
    hits2 = annif.hit.AnalysisResult([
        annif.hit.AnalysisHit(
            uri='http://www.yso.fi/onto/yso/p1747',
            label='egyptologit',
            score=1.0)])
    batch.evaluate(hits2, gold_set)
    results = batch.results()
    assert results['Precision (per document average)'] == 0.5
    assert results['Recall (per document average)'] == 0.5
    assert results['Label ranking average precision'] >= 0.50
    assert results['Label ranking average precision'] <= 0.51
    assert results['True positives'] == 1
    assert results['False positives'] == 1
    assert results['False negatives'] == 1
