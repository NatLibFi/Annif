"""Evaluation metrics for Annif"""


def precision(selected, relevant):
    """return the precision, i.e. the fraction of selected instances that are relevant"""
    if len(selected) == 0:
        return 0.0  # avoid division by zero
    return len(selected & relevant) / len(selected)


def recall(selected, relevant):
    """return the recall, i.e. the fraction of relevant instances that were selected"""
    if len(relevant) == 0:
        return 0.0  # avoid division by zero
    return len(selected & relevant) / len(relevant)


def f_measure(setA, setB):
    """return the F-measure similarity of two sets"""
    if len(setA) == 0 or len(setB) == 0:
        return 0.0  # shortcut, avoid division by zero
    return 2.0 * len(setA & setB) / (len(setA) + len(setB))


def evaluate(selected, gold):
    """evaluate a set of selected subject against a gold standard using different metrics"""
    return [
        ('Precision', precision(selected, gold)),
        ('Recall', recall(selected, gold)),
        ('F-measure', f_measure(selected, gold))
    ]
