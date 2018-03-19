"""Unit tests for corpus functionality in Annif"""

import annif.analyzer
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


def test_subjdir(tmpdir):
    tmpdir.join('subj1.txt').write("""http://example.org/subj1 subject one
        first subject
        this is the first thing we know about""")
    tmpdir.join('subj2.txt').write("""http://example.org/subj2 subject two
        second subject
        this is the second thing we know about""")
    tmpdir.join('subj3.txt').write("""http://example.org/subj3 subject three
        third subject
        this is the third thing we know about""")

    subjdir = annif.corpus.SubjectDirectory(str(tmpdir))
    subjects = sorted(list(subjdir), key=lambda subj: subj.uri)
    assert len(subjects) == 3
    assert subjects[0].uri == 'http://example.org/subj1'
    assert subjects[0].label == 'subject one'
    assert 'first' in subjects[0].text
    assert subjects[1].uri == 'http://example.org/subj2'
    assert subjects[1].label == 'subject two'
    assert 'second' in subjects[1].text
    assert subjects[2].uri == 'http://example.org/subj3'
    assert subjects[2].label == 'subject three'
    assert 'third' in subjects[2].text


def test_subjdir_texts(tmpdir):
    tmpdir.join('subj1.txt').write("""http://example.org/subj1 subject one
        first subject
        this is the first thing we know about""")
    tmpdir.join('subj2.txt').write("""http://example.org/subj2 subject two
        second subject
        this is the second thing we know about""")
    tmpdir.join('subj3.txt').write("""http://example.org/subj3 subject three
        third subject
        this is the third thing we know about""")

    subjdir = annif.corpus.SubjectDirectory(str(tmpdir))
    analyzer = annif.analyzer.get_analyzer('snowball(english)')
    tokens = list(subjdir.tokens(analyzer))
    assert 'first' in tokens[0]
    assert 'second' in tokens[1]
    assert 'third' in tokens[2]
