"""Unit tests for SKOS vocabulary functionality in Annif"""


from annif.corpus.skos import SubjectFileSKOS


def test_recognize_turtle():
    assert SubjectFileSKOS.is_rdf_file('subjects.ttl')


def test_recognize_rdfxml():
    assert SubjectFileSKOS.is_rdf_file('subjects.rdf')


def test_recognize_nt():
    assert SubjectFileSKOS.is_rdf_file('subjects.nt')


def test_recognize_tsv():
    assert not SubjectFileSKOS.is_rdf_file('subjects.tsv')


def test_recognize_noext():
    assert not SubjectFileSKOS.is_rdf_file('subjects')


def test_load_turtle(tmpdir):
    tmpfile = tmpdir.join('subjects.ttl')
    tmpfile.write("""
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
    """)

    corpus = SubjectFileSKOS(str(tmpfile), 'fi')
    subjects = list(corpus.subjects)
    assert len(subjects) == 1  # one of the concepts was deprecated
    assert subjects[0].uri == 'http://www.yso.fi/onto/yso/p8993'
    assert subjects[0].label == 'hylyt'
