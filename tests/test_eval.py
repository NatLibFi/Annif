"""Unit tests for evaluation metrics in Annif"""

import annif.eval


def test_precision():
    selected = set(['A', 'C', 'E', 'F', 'H'])
    gold = set(['A', 'B', 'C', 'D'])
    assert annif.eval.precision(selected, gold) == 0.4


def test_precision_empty():
    selected = set()
    gold = set(['A', 'B', 'C', 'D'])
    assert annif.eval.precision(selected, gold) == 0.0


def test_precision_empty2():
    selected = set(['A', 'C', 'E', 'F', 'H'])
    gold = set()
    assert annif.eval.precision(selected, gold) == 0.0


def test_recall():
    selected = set(['A', 'C', 'E', 'F'])
    gold = set(['A', 'B', 'C', 'D', 'E'])
    assert annif.eval.recall(selected, gold) == 0.6


def test_recall_empty():
    selected = set()
    gold = set(['A', 'B', 'C', 'D', 'E'])
    assert annif.eval.recall(selected, gold) == 0.0


def test_recall_empty2():
    selected = set(['A', 'C', 'E', 'F'])
    gold = set()
    assert annif.eval.recall(selected, gold) == 0.0


def test_f_measure():
    selected = set(['complex systems', 'network', 'small world'])
    gold = set(['theoretical', 'small world', 'network', 'dynamics'])
    f_measure = annif.eval.f_measure(selected, gold)
    assert f_measure > 0.57
    assert f_measure < 0.58


def test_ndcg_empty():
    selected = []
    gold = ['A', 'E', 'I', 'O', 'U']
    ndcg = annif.eval.normalized_dcg(selected, gold, 5)
    assert ndcg == 0


def test_ndcg_empty2():
    selected = ['A', 'B', 'C', 'D', 'E']
    gold = []
    ndcg = annif.eval.normalized_dcg(selected, gold, 5)
    assert ndcg == 0


def test_ndcg_5():
    selected = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # len=7
    gold = ['A', 'E', 'I', 'O', 'U', 'Z', 'X',
            'C', 'V', 'B', 'N', 'M']  # len=12
    ndcg = annif.eval.normalized_dcg(selected, gold, 5)
    assert ndcg > 0.85
    assert ndcg < 0.86


def test_ndcg_10():
    selected = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # len=7
    gold = ['A', 'E', 'I', 'O', 'U', 'Z', 'X',
            'C', 'V', 'B', 'N', 'M']  # len=12
    ndcg = annif.eval.normalized_dcg(selected, gold, 10)
    assert ndcg > 0.55
    assert ndcg < 0.56


def test_true_positives():
    selected = ['A', 'B', 'C']
    gold = ['A', 'C', 'E', 'F', 'G']
    true_positives = annif.eval.true_positives(selected, gold)
    assert true_positives == 2


def test_false_positives():
    selected = ['A', 'B', 'C']
    gold = ['A', 'C', 'E', 'F', 'G']
    false_positives = annif.eval.false_positives(selected, gold)
    assert false_positives == 1


def test_false_negatives():
    selected = ['A', 'B', 'C']
    gold = ['A', 'C', 'E', 'F', 'G']
    false_negatives = annif.eval.false_negatives(selected, gold)
    assert false_negatives == 3
