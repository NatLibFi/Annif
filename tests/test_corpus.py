"""Unit tests for corpus functionality in Annif"""

import annif.corpus


def test_subjectset_uris():
    data = """<http://example.org/dummy>\tdummy
    <http://example.org/another>\tanother
    """

    sset = annif.corpus.SubjectSet(data)
    assert sset.has_uris()
    assert len(sset.subject_uris) == 2
    assert "http://example.org/dummy" in sset.subject_uris
    assert "http://example.org/another" in sset.subject_uris


def test_subjectset_labels():
    data = """dummy
    another
    """

    sset = annif.corpus.SubjectSet(data)
    assert not sset.has_uris()
    assert len(sset.subject_labels) == 2
    assert "dummy" in sset.subject_labels
    assert "another" in sset.subject_labels
