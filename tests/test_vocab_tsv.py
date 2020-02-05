"""Unit tests for TSV vocabulary functionality in Annif"""


from annif.corpus import SubjectIndex


def test_load_tsv_uri_brackets(tmpdir):
    tmpfile = tmpdir.join('subjects.tsv')
    tmpfile.write("<http://www.yso.fi/onto/yso/p8993>\thylyt\n" +
                  "<http://www.yso.fi/onto/yso/p9285>\tneoliittinen kausi")

    index = SubjectIndex.load(str(tmpfile))
    assert len(index) == 2
    assert index[0] == ('http://www.yso.fi/onto/yso/p8993', 'hylyt', None)
    assert index[1] == (
        'http://www.yso.fi/onto/yso/p9285',
        'neoliittinen kausi', None)


def test_load_tsv_uri_nobrackets(tmpdir):
    tmpfile = tmpdir.join('subjects.tsv')
    tmpfile.write("http://www.yso.fi/onto/yso/p8993\thylyt\n" +
                  "http://www.yso.fi/onto/yso/p9285\tneoliittinen kausi")

    index = SubjectIndex.load(str(tmpfile))
    assert len(index) == 2
    assert index[0] == ('http://www.yso.fi/onto/yso/p8993', 'hylyt', None)
    assert index[1] == (
        'http://www.yso.fi/onto/yso/p9285',
        'neoliittinen kausi', None)


def test_load_tsv_with_notations(tmpdir):
    tmpfile = tmpdir.join('subjects-with-notations.tsv')
    tmpfile.write("http://www.yso.fi/onto/yso/p8993\thylyt\t42.42\n" +
                  "http://www.yso.fi/onto/yso/p9285\tneoliittinen kausi\t42.0")

    index = SubjectIndex.load(str(tmpfile))
    assert len(index) == 2
    assert index[0] == ('http://www.yso.fi/onto/yso/p8993', 'hylyt', '42.42')
    assert index[1] == (
        'http://www.yso.fi/onto/yso/p9285',
        'neoliittinen kausi', '42.0')
