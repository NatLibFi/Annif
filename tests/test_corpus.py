"""Unit tests for corpus functionality in Annif"""

import gzip
import numpy as np
import pytest
import annif.corpus
from annif.corpus import TransformingDocumentCorpus


def test_subjectset_uris(subject_index):
    data = """<http://www.yso.fi/onto/yso/p2558>\trautakausi
    <http://www.yso.fi/onto/yso/p12738>\tviikinkiaika
    """

    sset = annif.corpus.SubjectSet.from_string(data, subject_index)
    assert len(sset) == 2
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p2558") in sset
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p12738") in sset


def test_subjectset_labels(subject_index):
    data = """rautakausi
    viikinkiaika
    """

    sset = annif.corpus.SubjectSet.from_string(data, subject_index)
    assert len(sset) == 2
    assert subject_index.by_label("rautakausi") in sset
    assert subject_index.by_label("viikinkiaika") in sset


def test_subjectset_from_list(subject_index):
    uris = ['http://www.yso.fi/onto/yso/p10849',
            'http://www.yso.fi/onto/yso/p19740']
    subject_ids = [subject_index.by_uri(uri) for uri in uris]
    sset = annif.corpus.SubjectSet(subject_ids)
    assert len(sset) == 2
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p10849") in sset
    assert subject_index.by_uri("http://www.yso.fi/onto/yso/p19740") in sset


def test_subjectset_empty():
    sset = annif.corpus.SubjectSet()
    assert len(sset) == 0
    assert not sset
    with pytest.raises(IndexError):
        sset[0]


def test_subjectset_equal():
    sset = annif.corpus.SubjectSet([1, 3, 5])
    sset2 = annif.corpus.SubjectSet([3, 5, 1])
    assert sset == sset2


def test_subjectset_nonequal():
    sset = annif.corpus.SubjectSet([1, 3, 5])
    sset2 = annif.corpus.SubjectSet([3, 5])
    assert sset != sset2
    assert sset != [1, 3, 5]
    assert sset != set([1, 3, 5])


def test_subjectset_as_vector(subject_index):
    uris = ['http://www.yso.fi/onto/yso/p10849', 'http://example.org/unknown']
    subject_ids = [subject_index.by_uri(uri) for uri in uris]
    sset = annif.corpus.SubjectSet(subject_ids)
    vector = sset.as_vector(len(subject_index))
    assert vector.sum() == 1  # only one known subject


def test_subjectset_as_vector_destination(subject_index):
    uris = ['http://www.yso.fi/onto/yso/p10849', 'http://example.org/unknown']
    subject_ids = [subject_index.by_uri(uri) for uri in uris]
    sset = annif.corpus.SubjectSet(subject_ids)
    destination = np.zeros(len(subject_index), dtype=np.float32)
    vector = sset.as_vector(destination=destination)
    assert vector.sum() == 1  # only one known subject
    assert vector is destination


def test_docdir_key(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('key1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('key2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index)
    files = sorted(list(docdir))
    assert len(files) == 3
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.key'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.key'))
    assert files[2][0] == str(tmpdir.join('doc3.txt'))
    assert files[2][1] is None


def test_docdir_tsv(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.tsv').write('<http://example.org/key1>\tkey1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.tsv').write('<http://example.org/key2>\tkey2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index)
    files = sorted(list(docdir))
    assert len(files) == 3
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.tsv'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.tsv'))
    assert files[2][0] == str(tmpdir.join('doc3.txt'))
    assert files[2][1] is None


def test_docdir_tsv_bom(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1'.encode('utf-8-sig'))
    tmpdir.join('doc1.tsv').write(
        '<http://www.yso.fi/onto/yso/p4622>\tesihistoria'.encode('utf-8-sig'))
    tmpdir.join('doc2.txt').write('doc2'.encode('utf-8-sig'))
    tmpdir.join('doc2.tsv').write(
        '<http://www.yso.fi/onto/yso/p2558>\trautakausi'.encode('utf-8-sig'))

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index)
    docs = list(docdir.documents)
    assert docs[0].text == 'doc1'
    assert subject_index.by_uri(
        'http://www.yso.fi/onto/yso/p4622') in docs[0].subject_set
    assert len(docs[0].subject_set) == 1
    assert docs[1].text == 'doc2'
    assert subject_index.by_uri(
        'http://www.yso.fi/onto/yso/p2558') in docs[1].subject_set
    assert len(docs[1].subject_set) == 1


def test_docdir_key_require_subjects(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('<http://example.org/key1>\tkey1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('<http://example.org/key2>\tkey2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index,
                                            require_subjects=True)
    files = sorted(list(docdir))
    assert len(files) == 2
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.key'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.key'))


def test_docdir_tsv_require_subjects(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.tsv').write('key1')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.tsv').write('key2')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index,
                                            require_subjects=True)
    files = sorted(list(docdir))
    assert len(files) == 2
    assert files[0][0] == str(tmpdir.join('doc1.txt'))
    assert files[0][1] == str(tmpdir.join('doc1.tsv'))
    assert files[1][0] == str(tmpdir.join('doc2.txt'))
    assert files[1][1] == str(tmpdir.join('doc2.tsv'))


def test_docdir_tsv_as_doccorpus(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.tsv').write(
        '<http://www.yso.fi/onto/yso/p4622>\tesihistoria')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.tsv').write(
        '<http://www.yso.fi/onto/yso/p2558>\trautakausi')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index,
                                            require_subjects=True)
    docs = list(docdir.documents)
    assert len(docs) == 2
    assert docs[0].text == 'doc1'
    assert len(docs[0].subject_set) == 1
    assert subject_index.by_uri(
        'http://www.yso.fi/onto/yso/p4622') in docs[0].subject_set
    assert docs[1].text == 'doc2'
    assert subject_index.by_uri(
        'http://www.yso.fi/onto/yso/p2558') in docs[1].subject_set
    assert len(docs[1].subject_set) == 1


def test_docdir_key_as_doccorpus(tmpdir, subject_index):
    tmpdir.join('doc1.txt').write('doc1')
    tmpdir.join('doc1.key').write('arkeologit')
    tmpdir.join('doc2.txt').write('doc2')
    tmpdir.join('doc2.key').write('kalliotaide')
    tmpdir.join('doc3.txt').write('doc3')

    docdir = annif.corpus.DocumentDirectory(str(tmpdir), subject_index,
                                            require_subjects=True)
    docs = list(docdir.documents)
    assert len(docs) == 2
    assert docs[0].text == 'doc1'
    assert len(docs[0].subject_set) == 1
    assert subject_index.by_uri(
        'http://www.yso.fi/onto/yso/p10849') in docs[0].subject_set
    assert docs[1].text == 'doc2'
    assert len(docs[1].subject_set) == 1
    assert subject_index.by_uri(
        'http://www.yso.fi/onto/yso/p13027') in docs[1].subject_set


def test_subject_by_uri(subject_index):
    subj_id = subject_index.by_uri('http://www.yso.fi/onto/yso/p7141')
    assert subject_index[subj_id][1] == 'sinetit'


def test_subject_by_uri_missing(subject_index):
    subj_id = subject_index.by_uri('http://nonexistent')
    assert subj_id is None


def test_subject_by_label(subject_index):
    subj_id = subject_index.by_label('sinetit')
    assert subject_index[subj_id][0] == 'http://www.yso.fi/onto/yso/p7141'


def test_subject_by_label_missing(subject_index):
    subj_id = subject_index.by_label('nonexistent')
    assert subj_id is None


def test_docfile_plain(tmpdir, subject_index):
    docfile = tmpdir.join('documents.tsv')
    docfile.write("""Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>""")

    docs = annif.corpus.DocumentFile(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3


def test_docfile_bom(tmpdir, subject_index):
    docfile = tmpdir.join('documents_bom.tsv')
    data = """Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>"""
    docfile.write(data.encode('utf-8-sig'))

    docs = annif.corpus.DocumentFile(str(docfile), subject_index)
    firstdoc = next(docs.documents)
    assert firstdoc.text.startswith("Läntinen")


def test_docfile_plain_invalid_lines(tmpdir, caplog, subject_index):
    logger = annif.logger
    logger.propagate = True
    docfile = tmpdir.join('documents_invalid.tsv')
    docfile.write("""Läntinen\t<http://www.yso.fi/onto/yso/p2557>

        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        A line with no tabs
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>""")
    docs = annif.corpus.DocumentFile(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3
    assert len(caplog.records) == 2
    expected_msg = 'Skipping invalid line (missing tab):'
    for record in caplog.records:
        assert expected_msg in record.message


def test_docfile_gzipped(tmpdir, subject_index):
    docfile = tmpdir.join('documents.tsv.gz')
    with gzip.open(str(docfile), 'wt') as gzf:
        gzf.write("""Pohjoinen\t<http://www.yso.fi/onto/yso/p2557>
            Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
            Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>""")

    docs = annif.corpus.DocumentFile(str(docfile), subject_index)
    assert len(list(docs.documents)) == 3


def test_docfile_is_empty(tmpdir, subject_index):
    empty_file = tmpdir.ensure('empty.tsv')
    docs = annif.corpus.DocumentFile(str(empty_file), subject_index)
    assert docs.is_empty()


def test_combinedcorpus(tmpdir, subject_index):
    docfile = tmpdir.join('documents.tsv')
    docfile.write("""Läntinen\t<http://www.yso.fi/onto/yso/p2557>
        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>""")

    corpus1 = annif.corpus.DocumentFile(str(docfile), subject_index)
    corpus2 = annif.corpus.DocumentFile(str(docfile), subject_index)

    combined = annif.corpus.CombinedCorpus([corpus1, corpus2])

    assert len(list(combined.documents)) == 6


def test_transformingcorpus(document_corpus):
    def double(x): return x + x

    transformed_corpus = TransformingDocumentCorpus(document_corpus, double)
    for transf_doc, doc in zip(transformed_corpus.documents,
                               document_corpus.documents):
        assert transf_doc.text == doc.text + doc.text
        assert transf_doc.subject_set == doc.subject_set
    # Ensure docs are still available after iterating
    assert len(list(transformed_corpus.documents)) \
        == len(list(document_corpus.documents))


def test_limitingcorpus(tmpdir, subject_index):
    docfile = tmpdir.join('documents_invalid.tsv')
    docfile.write("""Läntinen\t<http://www.yso.fi/onto/yso/p2557>

        Oulunlinnan\t<http://www.yso.fi/onto/yso/p7346>
        A line with no tabs
        Harald Hirmuinen\t<http://www.yso.fi/onto/yso/p6479>""")

    document_corpus = annif.corpus.DocumentFile(str(docfile), subject_index)
    limiting_corpus = annif.corpus.LimitingDocumentCorpus(document_corpus, 2)

    assert len(list(limiting_corpus.documents)) == 2
    for limited_doc, doc in zip(limiting_corpus.documents,
                                document_corpus.documents):
        assert limited_doc.text == doc.text
