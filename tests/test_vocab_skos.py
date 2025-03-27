"""Unit tests for SKOS vocabulary functionality in Annif"""

import os.path

from annif.vocab import VocabFileSKOS


def test_recognize_turtle():
    assert VocabFileSKOS.is_rdf_file("subjects.ttl")


def test_recognize_rdfxml():
    assert VocabFileSKOS.is_rdf_file("subjects.rdf")


def test_recognize_nt():
    assert VocabFileSKOS.is_rdf_file("subjects.nt")


def test_recognize_tsv():
    assert not VocabFileSKOS.is_rdf_file("subjects.tsv")


def test_recognize_noext():
    assert not VocabFileSKOS.is_rdf_file("subjects")


def test_load_turtle(tmpdir):
    tmpfile = tmpdir.join("subjects.ttl")
    tmpfile.write(
        """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix yso: <http://www.yso.fi/onto/yso/> .
@prefix owl: <http://www.w3.org/2002/07/owl#>.

yso:p8993
    a skos:Concept ;
    skos:prefLabel "hylyt"@fi, "shipwrecks (objects)"@en, "vrak"@sv ;
    skos:related yso:p8869 .

yso:p9285
    a skos:Concept ;
    owl:deprecated true ;
    skos:broader yso:p4624 ;
    skos:prefLabel "Neolithic period"@en, "neoliittinen kausi"@fi,
        "neolitisk tid"@sv .
    """
    )

    corpus = VocabFileSKOS(str(tmpfile))
    subjects = list(corpus.subjects)
    assert len(subjects) == 1  # one of the concepts was deprecated
    assert subjects[0].uri == "http://www.yso.fi/onto/yso/p8993"
    assert subjects[0].labels["fi"] == "hylyt"
    assert subjects[0].notation is None


def test_load_turtle_with_notation(tmpdir):
    tmpfile = tmpdir.join("subjects.ttl")
    tmpfile.write(
        """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix yso: <http://www.yso.fi/onto/yso/> .
@prefix owl: <http://www.w3.org/2002/07/owl#>.

yso:p8993
    a skos:Concept ;
    skos:prefLabel "hylyt"@fi, "shipwrecks (objects)"@en, "vrak"@sv ;
    skos:notation "42.42" ;
    skos:related yso:p8869 .
    """
    )

    corpus = VocabFileSKOS(str(tmpfile))
    subjects = list(corpus.subjects)
    assert subjects[0].uri == "http://www.yso.fi/onto/yso/p8993"
    assert subjects[0].labels["fi"] == "hylyt"
    assert subjects[0].notation == "42.42"


def test_load_turtle_missing_langtags(tmpdir):
    tmpfile = tmpdir.join("subjects.ttl")
    tmpfile.write(
        """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex: <http://example.org/> .

# Concept with a prefLabel in English
ex:conc1 a skos:Concept;
    skos:prefLabel "Concept 1"@en .

# Concept with a prefLabel without a language tag
ex:conc2 a skos:Concept;
    skos:prefLabel "Concept 2" .

# Concept without a prefLabel
ex:conc3 a skos:Concept .
    """
    )

    corpus = VocabFileSKOS(str(tmpfile))
    subjects = list(corpus.subjects)
    assert len(subjects) == 3

    # check that the vocabulary contains the expected labels
    en_labels = {subj.labels["en"] for subj in subjects}
    assert "Concept 1" in en_labels
    assert "Concept 2" in en_labels
    assert "ex:conc3" in en_labels


def test_load_turtle_get_languages(testdatadir):
    subjectfile = os.path.join(
        os.path.dirname(__file__), "corpora", "archaeology", "yso-archaeology.ttl"
    )
    corpus = VocabFileSKOS(subjectfile)
    langs = corpus.languages
    assert len(langs) == 3
    assert "fi" in langs
    assert "sv" in langs
    assert "en" in langs
