"""Unit tests for TSV vocabulary functionality in Annif"""


from annif.corpus import SubjectIndex


def test_load_tsv_uri_brackets(tmpdir):
    tmpfile = tmpdir.join('subjects.tsv')
    tmpfile.write("<http://www.yso.fi/onto/yso/p8993>\thylyt\n" +
                  "<http://www.yso.fi/onto/yso/p9285>\tneoliittinen kausi")

    index = SubjectIndex.load(str(tmpfile))
    assert len(index) == 2
    assert index[0] == ('http://www.yso.fi/onto/yso/p8993', 'hylyt')
    assert index[1] == (
        'http://www.yso.fi/onto/yso/p9285',
        'neoliittinen kausi')


def test_load_tsv_uri_nobrackets(tmpdir):
    tmpfile = tmpdir.join('subjects.tsv')
    tmpfile.write("http://www.yso.fi/onto/yso/p8993\thylyt\n" +
                  "http://www.yso.fi/onto/yso/p9285\tneoliittinen kausi")

    index = SubjectIndex.load(str(tmpfile))
    assert len(index) == 2
    assert index[0] == ('http://www.yso.fi/onto/yso/p8993', 'hylyt')
    assert index[1] == (
        'http://www.yso.fi/onto/yso/p9285',
        'neoliittinen kausi')
