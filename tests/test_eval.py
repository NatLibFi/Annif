"""Unit tests for evaluation metrics in Annif"""

import annif.corpus
import annif.eval
import annif.hit


def test_ndcg_empty():
    selected = [[]]
    gold = [['A', 'E', 'I', 'O', 'U']]
    ndcg = annif.eval.normalized_dcg(selected, gold, 5)
    assert ndcg == 0


def test_ndcg_empty2():
    selected = [['A', 'B', 'C', 'D', 'E']]
    gold = [[]]
    ndcg = annif.eval.normalized_dcg(selected, gold, 5)
    assert ndcg == 0


def test_ndcg_5():
    selected = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]  # len=7
    gold = [['A', 'E', 'I', 'O', 'U', 'Z', 'X',
             'C', 'V', 'B', 'N', 'M']]  # len=12
    ndcg = annif.eval.normalized_dcg(selected, gold, 5)
    assert ndcg > 0.85
    assert ndcg < 0.86


def test_ndcg_10():
    selected = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]  # len=7
    gold = [['A', 'E', 'I', 'O', 'U', 'Z', 'X',
             'C', 'V', 'B', 'N', 'M']]  # len=12
    ndcg = annif.eval.normalized_dcg(selected, gold, 10)
    assert ndcg > 0.55
    assert ndcg < 0.56


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
