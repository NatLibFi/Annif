"""Unit tests for Annif utility functions"""

import os
from contextlib import contextmanager

import pytest

import annif.util


@contextmanager
def umask_context(mask):
    """Context manager to temporarily set umask"""
    original_umask = os.umask(0)  # Get current umask
    os.umask(mask)  # Set new umask
    try:
        yield
    finally:
        os.umask(original_umask)  # Restore original umask


class MockSaveable:
    def save(self, filename):
        with open(filename, "w") as f:
            f.write("data")


# parametrize test to verify that file permissions are set correctly
# using commonly used umask values
@pytest.mark.parametrize(
    "umask,mode", [(0o002, 0o664), (0o022, 0o644), (0o027, 0o640), (0o077, 0o600)]
)
def test_atomic_save(tmpdir, umask, mode):
    obj = MockSaveable()
    dirname = str(tmpdir)
    filename = "myfile"

    with umask_context(umask):
        annif.util.atomic_save(obj, dirname, filename)

    final_path = tmpdir.join(filename)
    assert final_path.exists()

    # verify file content
    assert final_path.read_text(encoding="utf-8") == "data"

    # verify file permissions
    assert final_path.stat().mode & 0o777 == mode


def test_boolean():
    inputs = ["1", "0", "true", "false", "TRUE", "FALSE", "Yes", "No", True, False]
    outputs = [True, False, True, False, True, False, True, False, True, False]

    for input, output in zip(inputs, outputs):
        assert annif.util.boolean(input) == output


def test_metric_code():
    inputs = ["F1 score (doc avg)", "NDCG@10", "True positives"]
    outputs = ["F1_score_doc_avg", "NDCG@10", "True_positives"]

    for input, output in zip(inputs, outputs):
        assert annif.util.metric_code(input) == output
