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
