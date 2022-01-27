"""Unit tests for Annif utility functions"""

import annif.util


def test_boolean():
    inputs = [
        '1',
        '0',
        'true',
        'false',
        'TRUE',
        'FALSE',
        'Yes',
        'No',
        True,
        False]
    outputs = [True, False, True, False, True, False, True, False, True, False]

    for input, output in zip(inputs, outputs):
        assert annif.util.boolean(input) == output


def test_metric_code():
    inputs = [
        "F1 score (doc avg)",
        "NDCG@10",
        "True positives"]
    outputs = [
        "F1_score_doc_avg",
        "NDCG@10",
        "True_positives"]

    for input, output in zip(inputs, outputs):
        assert annif.util.metric_code(input) == output
