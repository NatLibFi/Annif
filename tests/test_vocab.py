"""Unit tests for vocabulary functionality in Annif"""

import pytest
import os
import annif.corpus
import annif.vocab
from annif.exception import NotInitializedException
import rdflib.namespace


def load_dummy_vocab(tmpdir):
    vocab = annif.vocab.AnnifVocabulary('vocab-id', str(tmpdir), 'en')
    subjfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'dummy-subjects.tsv')
    subjects = annif.corpus.SubjectFileTSV(subjfile, 'en')
    vocab.load_vocabulary(subjects)
    return vocab


def test_get_vocab_invalid(registry):
    with pytest.raises(ValueError) as excinfo:
        registry.get_vocab('', None)
    assert 'Invalid vocabulary specification' in str(excinfo.value)


def test_update_subject_index_with_no_changes(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'dummy-subjects.tsv')
    subjects = annif.corpus.SubjectFileTSV(subjfile, 'en')

    vocab.load_vocabulary(subjects)
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0].uri == 'http://example.org/dummy'
    assert vocab.subjects[0].labels['en'] == 'dummy'
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1].uri == 'http://example.org/none'
    assert vocab.subjects[1].labels['en'] == 'none'
    assert vocab.subjects[1].notation == '42.42'


def test_update_subject_index_with_removed_subject(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new), 'en')

    vocab.load_vocabulary(subjects_new)
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0].uri == 'http://example.org/dummy'
    assert vocab.subjects[0].labels['en'] == 'dummy'
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1].uri == 'http://example.org/none'
    assert vocab.subjects[1].labels is None
    assert vocab.subjects[1].notation is None


def test_update_subject_index_with_renamed_label_and_added_notation(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n" +
                       "<http://example.org/none>\tnew none\t42.42\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new), 'en')

    vocab.load_vocabulary(subjects_new)
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0].uri == 'http://example.org/dummy'
    assert vocab.subjects[0].labels['en'] == 'dummy'
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri('http://example.org/none') == 1
    assert vocab.subjects[1].uri == 'http://example.org/none'
    assert vocab.subjects[1].labels['en'] == 'new none'
    assert vocab.subjects[1].notation == '42.42'


def test_update_subject_index_with_added_subjects(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n" +
                       "<http://example.org/none>\tnone\n" +
                       "<http://example.org/new-dummy>\tnew dummy\t42.42\n" +
                       "<http://example.org/new-none>\tnew none\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new), 'en')

    vocab.load_vocabulary(subjects_new)
    assert len(vocab.subjects) == 4
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0].uri == 'http://example.org/dummy'
    assert vocab.subjects[0].labels['en'] == 'dummy'
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri('http://example.org/new-dummy') == 2
    assert vocab.subjects[2].uri == 'http://example.org/new-dummy'
    assert vocab.subjects[2].labels['en'] == 'new dummy'
    assert vocab.subjects[2].notation == '42.42'


def test_update_subject_index_force(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    subjfile_new = tmpdir.join('subjects_new.tsv')
    subjfile_new.write("<http://example.org/dummy>\tdummy\n" +
                       "<http://example.org/new-dummy>\tnew dummy\t42.42\n" +
                       "<http://example.org/new-none>\tnew none\n")
    subjects_new = annif.corpus.SubjectFileTSV(str(subjfile_new), 'en')

    vocab.load_vocabulary(subjects_new, force=True)
    assert len(vocab.subjects) == 3
    assert vocab.subjects.by_uri('http://example.org/dummy') == 0
    assert vocab.subjects[0].uri == 'http://example.org/dummy'
    assert vocab.subjects[0].labels['en'] == 'dummy'
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri('http://example.org/new-dummy') == 1
    assert vocab.subjects[1].uri == 'http://example.org/new-dummy'
    assert vocab.subjects[1].labels['en'] == 'new dummy'
    assert vocab.subjects[1].notation == '42.42'


def test_skos(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    assert tmpdir.join('vocabs/vocab-id/subjects.ttl').exists()
    assert tmpdir.join('vocabs/vocab-id/subjects.dump.gz').exists()
    assert isinstance(vocab.skos, annif.corpus.SubjectFileSKOS)


def test_skos_cache(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    assert tmpdir.join('vocabs/vocab-id/subjects.ttl').exists()
    assert tmpdir.join('vocabs/vocab-id/subjects.dump.gz').exists()
    tmpdir.join('vocabs/vocab-id/subjects.dump.gz').remove()
    assert not tmpdir.join('vocabs/vocab-id/subjects.dump.gz').exists()

    assert isinstance(vocab.skos, annif.corpus.SubjectFileSKOS)
    # cached dump file has been recreated in .skos property access
    assert tmpdir.join('vocabs/vocab-id/subjects.dump.gz').exists()


def test_skos_not_found(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    assert tmpdir.join('vocabs/vocab-id/subjects.ttl').exists()
    assert tmpdir.join('vocabs/vocab-id/subjects.dump.gz').exists()
    tmpdir.join('vocabs/vocab-id/subjects.ttl').remove()
    tmpdir.join('vocabs/vocab-id/subjects.dump.gz').remove()

    with pytest.raises(NotInitializedException):
        vocab.skos


def test_as_graph(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    graph = vocab.as_graph()
    labels = [
        (str(tpl[0]), str(tpl[1]))
        for tpl
        in graph[
            :rdflib.namespace.SKOS.prefLabel:]
    ]
    assert len(labels) == 2
    assert ('http://example.org/dummy',	'dummy') in labels
    assert ('http://example.org/none',	'none') in labels
    concepts = [
        str(tpl)
        for tpl
        in graph[
            :rdflib.namespace.RDF.type:rdflib.namespace.SKOS.Concept]
    ]
    assert len(concepts) == 2
    assert 'http://example.org/dummy' in concepts
    assert 'http://example.org/none' in concepts
