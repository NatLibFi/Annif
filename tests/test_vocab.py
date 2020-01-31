"""Unit tests for vocabulary functionality in Annif"""

import os
import annif.corpus
import annif.vocab


def test_update_subject_index(testdatadir, tmpdir, subject_index):
    subjfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'dummy-subjects.tsv')
    vocab = annif.vocab.AnnifVocabulary('vocab-id', str(tmpdir))
    subjects = annif.corpus.SubjectFileTSV(subjfile)

    # Load vocabulary first time
    vocab.load_vocabulary(subjects, 'en')

    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy')

    # Load the same vocabulary again, nothing changes
    vocab.load_vocabulary(subjects, 'en')
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy')

    # Update with new vocabulary
    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n" +
                       "<http://example.org/dummydummy>\tdummydummy\n" +
                       "<http://example.org/nonenone>\tnonenone\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new))
    vocab.load_vocabulary(subjects_new, 'en')
    assert len(vocab.subjects) == 4
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy')
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1] == ('http://example.org/none', '')
    assert vocab.subjects.by_uri('http://example.org/dummydummy') == 2
    assert vocab.subjects[2] == ('http://example.org/dummydummy', 'dummydummy')

    # Update by loading the original back
    vocab.load_vocabulary(subjects, 'en')
    assert len(vocab.subjects) == 4
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0] == ('http://example.org/dummy', 'dummy')
    assert vocab.subjects.by_uri('http://example.org/dummydummy') == 2
    assert vocab.subjects[2] == ('http://example.org/dummydummy', '')
