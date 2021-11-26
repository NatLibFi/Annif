"""Unit tests for Annif utility functions"""

import annif.util
from unittest.mock import MagicMock


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


def test_apply_parse_param_config():
    fun0 = MagicMock()
    fun0.return_value = 23
    fun1 = MagicMock()
    fun1.return_value = 'ret'
    configs = {
        'a': fun0,
        'c': fun1
    }
    params = {
        'a': 0,
        'b': 23,
        'c': None
    }
    ret = annif.util.apply_param_parse_config(configs, params)
    assert ret == {
        'a': 23
    }
    fun0.assert_called_once_with(0)
    fun1.assert_not_called()
