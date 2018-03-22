"""Evaluation metrics for Annif"""


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


def evaluate(selected, gold):
    """evaluate a set of selected subject against a gold standard using
    different metrics"""
    return [
        ('Precision', precision(selected, gold)),
        ('Recall', recall(selected, gold)),
        ('F-measure', f_measure(selected, gold)),
        ('Precision@1', precision(selected[:1], gold)),
        ('Precision@3', precision(selected[:3], gold)),
        ('Precision@5', precision(selected[:5], gold))
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
