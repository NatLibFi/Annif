"""Unit tests for evaluation metrics in Annif"""

from scipy.sparse import csr_array

import annif.corpus
import annif.eval
import annif.suggestion


def test_true_positives():
    y_true = csr_array([[True, False, True, False, True, False]])
    y_pred = csr_array([[True, True, False, True, True, False]])
    tp = annif.eval.true_positives(y_true, y_pred)
    assert tp == 2

    y_true = csr_array([[True, True, False, True, True, False]])
    y_pred = csr_array([[True, False, True, True, True, False]])
    tp2 = annif.eval.true_positives(y_true, y_pred)
    assert tp2 == 3


def test_false_positives():
    y_true = csr_array([[True, False, True, False, True, False]])
    y_pred = csr_array([[True, True, False, True, True, False]])
    fp = annif.eval.false_positives(y_true, y_pred)
    assert fp == 2

    y_true = csr_array([[True, True, False, True, True, False]])
    y_pred = csr_array([[True, False, True, True, True, False]])
    fp2 = annif.eval.false_positives(y_true, y_pred)
    assert fp2 == 1


def test_false_negatives():
    y_true = csr_array([[True, False, True, False, True, False]])
    y_pred = csr_array([[True, True, False, True, True, False]])
    fn = annif.eval.false_negatives(y_true, y_pred)
    assert fn == 1

    y_true = csr_array([[True, True, False, True, True, False]])
    y_pred = csr_array([[True, False, True, True, False, False]])
    fn2 = annif.eval.false_negatives(y_true, y_pred)
    assert fn2 == 2


# DCG@6 example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
def test_dcg():
    y_true = csr_array([[3, 2, 3, 0, 1, 2]])
    y_pred = csr_array([[6, 5, 4, 3, 2, 1]])
    dcg = annif.eval.dcg_score(y_true, y_pred, 6)
    assert dcg > 6.86
    assert dcg < 6.87


# iDCG@6 example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
def test_dcg_ideal():
    y_true = csr_array([[3, 3, 3, 2, 2, 2, 1, 0]])
    y_pred = csr_array([[8, 7, 6, 5, 4, 3, 2, 1]])
    dcg = annif.eval.dcg_score(y_true, y_pred, 6)
    assert dcg > 8.74
    assert dcg < 8.75


# nDCG@6 example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
def test_ndcg():
    y_true = csr_array([[3, 2, 3, 0, 1, 2, 3, 2]])
    y_pred = csr_array([[6, 5, 4, 3, 2, 1, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred, 6)
    assert ndcg > 0.78
    assert ndcg < 0.79


def test_ndcg_nolimit():
    y_true = csr_array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    y_pred = csr_array([[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred)
    assert ndcg > 0.49
    assert ndcg < 0.50


def test_ndcg_10():
    y_true = csr_array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    y_pred = csr_array([[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred, 10)
    assert ndcg > 0.55
    assert ndcg < 0.56


def test_ndcg_5():
    y_true = csr_array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    y_pred = csr_array([[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred, 5)
    assert ndcg > 0.85
    assert ndcg < 0.86


def test_ndcg_empty():
    y_true = csr_array([[1, 1, 1, 1, 1]])
    y_pred = csr_array([[0, 0, 0, 0, 0]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred)
    assert ndcg == 0


def test_ndcg_empty2():
    y_true = csr_array([[0, 0, 0, 0, 0]])
    y_pred = csr_array([[1, 1, 1, 1, 1]])
    ndcg = annif.eval.ndcg_score(y_true, y_pred)
    assert ndcg == 1.0


def test_evaluation_batch(subject_index, tmpdir):
    batch = annif.eval.EvaluationBatch(subject_index)

    gold_set = annif.corpus.SubjectSet.from_string(
        "<http://www.yso.fi/onto/yso/p10849>\tarkeologit", subject_index, "fi"
    )
    hits1 = [
        # subject: archaeologists (yso:p10849)
        annif.suggestion.SubjectSuggestion(
            subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p10849"),
            score=1.0,
        )
    ]
    batch.evaluate_many([hits1], [gold_set])
    hits2 = [
        # subject: egyptologists (yso:p1747)
        annif.suggestion.SubjectSuggestion(
            subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p1747"),
            score=1.0,
        )
    ]
    batch.evaluate_many([hits2], [gold_set])
    outfile = tmpdir.join("results.tsv")
    results = batch.results(results_file=outfile.open("w"), language="en")
    assert results["Precision (doc avg)"] == 0.5
    assert results["Recall (doc avg)"] == 0.5
    assert results["True positives"] == 1
    assert results["False positives"] == 1
    assert results["False negatives"] == 1
    assert results["Documents evaluated"] == 2

    output = outfile.readlines()
    assert len(output) == 131
    assert (
        output[0]
        == "\t".join(
            [
                "URI",
                "Label",
                "Support",
                "True_positives",
                "False_positives",
                "False_negatives",
                "Precision",
                "Recall",
                "F1_score",
            ]
        )
        + "\n"
    )
