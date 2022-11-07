"""Unit tests for TSV vocabulary functionality in Annif"""


from annif.corpus import SubjectFileTSV, SubjectIndex


def test_load_tsv_uri_brackets(tmpdir):
    tmpfile = tmpdir.join("subjects.tsv")
    tmpfile.write(
        "<http://www.yso.fi/onto/yso/p8993>\thylyt\n"
        + "<http://www.yso.fi/onto/yso/p9285>\tneoliittinen kausi"
    )

    corpus = SubjectFileTSV(str(tmpfile), "fi")
    subjects = list(corpus.subjects)
    assert len(subjects) == 2
    assert subjects[0].uri == "http://www.yso.fi/onto/yso/p8993"
    assert subjects[0].labels["fi"] == "hylyt"
    assert subjects[0].notation is None
    assert subjects[1].uri == "http://www.yso.fi/onto/yso/p9285"
    assert subjects[1].labels["fi"] == "neoliittinen kausi"
    assert subjects[1].notation is None


def test_load_tsv_uri_nobrackets(tmpdir):
    tmpfile = tmpdir.join("subjects.tsv")
    tmpfile.write(
        "http://www.yso.fi/onto/yso/p8993\thylyt\n"
        + "http://www.yso.fi/onto/yso/p9285\tneoliittinen kausi"
    )

    corpus = SubjectFileTSV(str(tmpfile), "fi")
    subjects = list(corpus.subjects)
    assert len(subjects) == 2
    assert subjects[0].uri == "http://www.yso.fi/onto/yso/p8993"
    assert subjects[0].labels["fi"] == "hylyt"
    assert subjects[0].notation is None
    assert subjects[1].uri == "http://www.yso.fi/onto/yso/p9285"
    assert subjects[1].labels["fi"] == "neoliittinen kausi"
    assert subjects[1].notation is None


def test_load_tsv_with_notations(tmpdir):
    tmpfile = tmpdir.join("subjects-with-notations.tsv")
    tmpfile.write(
        "http://www.yso.fi/onto/yso/p8993\thylyt\t42.42\n"
        + "http://www.yso.fi/onto/yso/p9285\tneoliittinen kausi\t42.0"
    )

    corpus = SubjectFileTSV(str(tmpfile), "fi")
    subjects = list(corpus.subjects)
    assert len(subjects) == 2
    assert subjects[0].uri == "http://www.yso.fi/onto/yso/p8993"
    assert subjects[0].labels["fi"] == "hylyt"
    assert subjects[0].notation == "42.42"
    assert subjects[1].uri == "http://www.yso.fi/onto/yso/p9285"
    assert subjects[1].labels["fi"] == "neoliittinen kausi"
    assert subjects[1].notation == "42.0"


def test_load_tsv_with_deprecated(tmpdir):
    tmpfile = tmpdir.join("subjects.tsv")
    tmpfile.write(
        "<http://www.yso.fi/onto/yso/p8993>\thylyt\n"
        + "<http://example.org/deprecated>\t\n"
        + "<http://www.yso.fi/onto/yso/p9285>\tneoliittinen kausi"
    )

    corpus = SubjectFileTSV(str(tmpfile), "fi")
    subjects = list(corpus.subjects)
    assert len(list(corpus.subjects)) == 3
    assert subjects[1].labels is None

    index = SubjectIndex()
    index.load_subjects(corpus)

    active = list(index.active)
    assert len(active) == 2
    assert active[0][1].uri == "http://www.yso.fi/onto/yso/p8993"
    assert active[0][1].labels["fi"] == "hylyt"
    assert active[0][1].notation is None
    assert active[1][1].uri == "http://www.yso.fi/onto/yso/p9285"
    assert active[1][1].labels["fi"] == "neoliittinen kausi"
    assert active[1][1].notation is None
