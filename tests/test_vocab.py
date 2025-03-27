"""Unit tests for vocabulary functionality in Annif"""

import os

import pytest
import rdflib.namespace

import annif.corpus
import annif.vocab
from annif.exception import NotInitializedException


def load_dummy_vocab(tmpdir):
    vocab = annif.vocab.AnnifVocabulary("vocab-id", str(tmpdir))
    subjfile = os.path.join(os.path.dirname(__file__), "corpora", "dummy-subjects.tsv")
    subjects = annif.vocab.VocabFileTSV(subjfile, "en")
    vocab.load_vocabulary(subjects)
    return vocab


def test_get_vocab_invalid(registry):
    with pytest.raises(ValueError) as excinfo:
        registry.get_vocab("")
    assert "Invalid vocabulary ID" in str(excinfo.value)


def test_get_vocab_hyphen(registry):
    vocab = registry.get_vocab("dummy-noname")
    assert vocab.vocab_id == "dummy-noname"
    assert vocab is not None


def test_update_subject_index_with_no_changes(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile = os.path.join(os.path.dirname(__file__), "corpora", "dummy-subjects.tsv")
    subjects = annif.vocab.VocabFileTSV(subjfile, "en")

    vocab.load_vocabulary(subjects)
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri("http://example.org/dummy") == 0
    assert vocab.subjects[0].uri == "http://example.org/dummy"
    assert vocab.subjects[0].labels["en"] == "dummy"
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri("http://example.org/none") == 1
    assert vocab.subjects[1].uri == "http://example.org/none"
    assert vocab.subjects[1].labels["en"] == "none"
    assert vocab.subjects[1].notation == "42.42"


def test_update_subject_index_with_removed_subject(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile_new = tmpdir.join("subjects_new.tsv")
    subjfile_new.write("<http://example.org/dummy>\tdummy\n")
    subjects_new = annif.vocab.VocabFileTSV(str(subjfile_new), "en")

    vocab.load_vocabulary(subjects_new)
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri("http://example.org/dummy") == 0
    assert vocab.subjects[0].uri == "http://example.org/dummy"
    assert vocab.subjects[0].labels["en"] == "dummy"
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri("http://example.org/none") is None


def test_update_subject_index_with_renamed_label_and_added_notation(tmpdir):
    vocab = load_dummy_vocab(tmpdir)

    subjfile_new = tmpdir.join("subjects_new.tsv")
    subjfile_new.write(
        "<http://example.org/dummy>\tdummy\n"
        + "<http://example.org/none>\tnew none\t42.42\n"
    )
    subjects_new = annif.vocab.VocabFileTSV(str(subjfile_new), "en")

    vocab.load_vocabulary(subjects_new)
    assert len(vocab.subjects) == 2
    assert vocab.subjects.by_uri("http://example.org/dummy") == 0
    assert vocab.subjects[0].uri == "http://example.org/dummy"
    assert vocab.subjects[0].labels["en"] == "dummy"
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri("http://example.org/none") == 1
    assert vocab.subjects[1].uri == "http://example.org/none"
    assert vocab.subjects[1].labels["en"] == "new none"
    assert vocab.subjects[1].notation == "42.42"


def test_update_subject_index_with_added_subjects(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    subjfile_new = tmpdir.join("subjects_new.tsv")
    subjfile_new.write(
        "<http://example.org/dummy>\tdummy\n"
        + "<http://example.org/none>\tnone\n"
        + "<http://example.org/new-dummy>\tnew dummy\t42.42\n"
        + "<http://example.org/new-none>\tnew none\n"
    )
    subjects_new = annif.vocab.VocabFileTSV(str(subjfile_new), "en")

    vocab.load_vocabulary(subjects_new)
    assert len(vocab.subjects) == 4
    assert vocab.subjects.by_uri("http://example.org/dummy") == 0
    assert vocab.subjects[0].uri == "http://example.org/dummy"
    assert vocab.subjects[0].labels["en"] == "dummy"
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri("http://example.org/new-dummy") == 2
    assert vocab.subjects[2].uri == "http://example.org/new-dummy"
    assert vocab.subjects[2].labels["en"] == "new dummy"
    assert vocab.subjects[2].notation == "42.42"


def test_update_subject_index_force(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    subjfile_new = tmpdir.join("subjects_new.tsv")
    subjfile_new.write(
        "<http://example.org/dummy>\tdummy\n"
        + "<http://example.org/new-dummy>\tnew dummy\t42.42\n"
        + "<http://example.org/new-none>\tnew none\n"
    )
    subjects_new = annif.vocab.VocabFileTSV(str(subjfile_new), "en")

    vocab.load_vocabulary(subjects_new, force=True)
    assert len(vocab.subjects) == 3
    assert vocab.subjects.by_uri("http://example.org/dummy") == 0
    assert vocab.subjects[0].uri == "http://example.org/dummy"
    assert vocab.subjects[0].labels["en"] == "dummy"
    assert vocab.subjects[0].notation is None
    assert vocab.subjects.by_uri("http://example.org/new-dummy") == 1
    assert vocab.subjects[1].uri == "http://example.org/new-dummy"
    assert vocab.subjects[1].labels["en"] == "new dummy"
    assert vocab.subjects[1].notation == "42.42"


def test_skos(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    assert tmpdir.join("vocabs/vocab-id/subjects.ttl").exists()
    assert tmpdir.join("vocabs/vocab-id/subjects.dump.gz").exists()
    assert isinstance(vocab.skos, annif.vocab.VocabFileSKOS)


def test_skos_cache(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    assert tmpdir.join("vocabs/vocab-id/subjects.ttl").exists()
    assert tmpdir.join("vocabs/vocab-id/subjects.dump.gz").exists()
    tmpdir.join("vocabs/vocab-id/subjects.dump.gz").remove()
    assert not tmpdir.join("vocabs/vocab-id/subjects.dump.gz").exists()

    assert isinstance(vocab.skos, annif.vocab.VocabFileSKOS)
    # cached dump file has been recreated in .skos property access
    assert tmpdir.join("vocabs/vocab-id/subjects.dump.gz").exists()


def test_skos_not_found(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    assert tmpdir.join("vocabs/vocab-id/subjects.ttl").exists()
    assert tmpdir.join("vocabs/vocab-id/subjects.dump.gz").exists()
    tmpdir.join("vocabs/vocab-id/subjects.ttl").remove()
    tmpdir.join("vocabs/vocab-id/subjects.dump.gz").remove()

    with pytest.raises(NotInitializedException):
        vocab.skos


def test_as_graph(tmpdir):
    vocab = load_dummy_vocab(tmpdir)
    graph = vocab.as_graph()
    labels = [
        (str(tpl[0]), str(tpl[1])) for tpl in graph[: rdflib.namespace.SKOS.prefLabel :]
    ]
    assert len(labels) == 2
    assert ("http://example.org/dummy", "dummy") in labels
    assert ("http://example.org/none", "none") in labels
    concepts = [
        str(tpl)
        for tpl in graph[: rdflib.namespace.RDF.type : rdflib.namespace.SKOS.Concept]
    ]
    assert len(concepts) == 2
    assert "http://example.org/dummy" in concepts
    assert "http://example.org/none" in concepts


def test_subject_by_uri(subject_index):
    subj_id = subject_index.by_uri("http://www.yso.fi/onto/yso/p7141")
    assert subject_index[subj_id].labels["fi"] == "sinetit"


def test_subject_by_uri_missing(subject_index):
    subj_id = subject_index.by_uri("http://nonexistent")
    assert subj_id is None


def test_subject_by_label(subject_index):
    subj_id = subject_index.by_label("sinetit", "fi")
    assert subject_index[subj_id].uri == "http://www.yso.fi/onto/yso/p7141"


def test_subject_by_label_missing(subject_index):
    subj_id = subject_index.by_label("nonexistent", "fi")
    assert subj_id is None


def test_subject_index_filter(subject_index):
    subject_filter = annif.vocab.SubjectIndexFilter(
        subject_index, exclude=["http://www.yso.fi/onto/yso/p7141"]
    )

    assert len(subject_index) == len(subject_filter)

    assert subject_index.languages == subject_filter.languages

    subj_id = subject_index.by_uri("http://www.yso.fi/onto/yso/p7141")
    with pytest.raises(IndexError):
        subject_filter[subj_id]

    assert not subject_filter.contains_uri("http://www.yso.fi/onto/yso/p7141")

    assert subject_filter.by_uri("http://www.yso.fi/onto/yso/p7141") is None

    assert subject_filter.by_label("sinetit", "fi") is None

    assert len(subject_filter.active) == len(subject_index.active) - 1
