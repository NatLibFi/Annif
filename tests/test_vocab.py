"""Unit tests for vocabulary functionality in Annif"""

import os
import annif.corpus
import annif.vocab


def load_dummy_vocab(tmpdir):
    vocab = annif.vocab.AnnifVocabulary('vocab-id', str(tmpdir))
    subjfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'dummy-subjects.tsv')
    subjects = annif.corpus.SubjectFileTSV(subjfile)
    vocab.load_vocabulary(subjects, 'en')
    return vocab


def test_update_subject_index_with_no_changes(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'dummy-subjects.tsv')
    subjects = annif.corpus.SubjectFileTSV(subjfile)

    vocab.load_vocabulary(subjects, 'en')
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy', None)
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1] == ('http://example.org/none', 'none', None)


def test_update_subject_index_with_removed_subject(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new))

    vocab.load_vocabulary(subjects_new, 'en')
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy', None)
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1] == ('http://example.org/none', None, None)


def test_update_subject_index_with_renamed_label_and_added_notation(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n" +
                       "<http://example.org/none>\tnew none\t42.42\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new))

    vocab.load_vocabulary(subjects_new, 'en')
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy', None)
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1] == ('http://example.org/none', 'new none',
                                 '42.42')


def test_update_subject_index_with_added_subjects(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n" +
                       "<http://example.org/none>\tnone\n" +
                       "<http://example.org/new-dummy>\tnew dummy\t42.42\n" +
                       "<http://example.org/new-none>\tnew none\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new))

    vocab.load_vocabulary(subjects_new, 'en')
    assert len(vocab.subjects) == 4
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy', None)
    assert vocab.subjects.by_uri('http://example.org/new-dummy') == 2
    assert vocab.subjects[2] == ('http://example.org/new-dummy', 'new dummy',
                                 '42.42')
