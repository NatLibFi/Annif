"""Unit tests for Annif utility functions"""

import annif.util
from unittest.mock import MagicMock
import os.path as osp


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


def _save(obj, pth):
    with open(pth, 'w') as f:
        print('test file content', file=f)


def test_atomic_save_method(tmpdir):
    fname = 'tst_file_method.txt'
    annif.util.atomic_save(None, tmpdir.strpath, fname, method=_save)
    f_pth = tmpdir.join(fname)
    assert f_pth.exists()
    with f_pth.open() as f:
        assert f.readlines() == ['test file content\n']


def test_atomic_save(tmpdir):
    fname = 'tst_file_obj.txt'
    to_save = MagicMock()
    to_save.save.side_effect = lambda pth: _save(None, pth)
    annif.util.atomic_save(to_save, tmpdir.strpath, fname)
    f_pth = tmpdir.join(fname)
    assert f_pth.exists()
    with f_pth.open() as f:
        assert f.readlines() == ['test file content\n']
    to_save.save.assert_called_once()
    call_args = to_save.save.calls[0].args
    assert isinstance(call_args[0], MagicMock)
    assert call_args[1] != f_pth.strpath


def test_atomic_save_folder(tmpdir):
    folder_name = 'test_save'
    fname_0 = 'tst_file_0'
    fname_1 = 'tst_file_1'

    def save_folder(obj, pth):
        _save(None, osp.join(pth, fname_0))
        _save(None, osp.join(pth, fname_1))
    folder_path = tmpdir.join(folder_name)
    annif.util.atomic_save_folder(
        None,
        folder_path.strpath,
        method=save_folder)
    assert folder_path.exists()
    for f_name in [fname_0, fname_1]:
        f_pth = folder_path.join(f_name)
        assert f_pth.exists()
        with f_pth.open() as f:
            assert f.readlines() == ['test file content\n']
