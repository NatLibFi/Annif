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


def test_docdir_key(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('key1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('key2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir))
    files = sorted(list(docdir))
    assert len(files) == 3
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.key'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.key'))
    assert files[2][0] == str(tmpdir.join('doc3.txt'))
    assert files[2][1] is None


def test_docdir_tsv(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.tsv').write('<http://example.org/key1>\tkey1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.tsv').write('<http://example.org/key2>\tkey2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir))
    files = sorted(list(docdir))
    assert len(files) == 3
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.tsv'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.tsv'))
    assert files[2][0] == str(tmpdir.join('doc3.txt'))
    assert files[2][1] is None


def test_docdir_key_require_subjects(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('<http://example.org/key1>\tkey1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('<http://example.org/key2>\tkey2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=True)
    files = sorted(list(docdir))
    assert len(files) == 2
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.key'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.key'))


def test_docdir_tsv_require_subjects(tmpdir):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.tsv').write('key1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.tsv').write('key2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), require_subjects=True)
    files = sorted(list(docdir))
    assert len(files) == 2
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.tsv'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.tsv'))
