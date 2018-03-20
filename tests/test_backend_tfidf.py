"""Unit tests for the TF-IDF backend in Annif"""

import annif
import annif.backend
import annif.corpus
import os.path


def test_tfidf_load_subjects(tmpdir):
    annif.cxapp.app.config['DATADIR'] = str(tmpdir)
    tfidf_type = annif.backend.get_backend_type("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        config={})

    subjdir = os.path.join(
        os.path.dirname(__file__),
        'corpora',
        'archaeology',
        'subjects')
    subjects = annif.corpus.SubjectDirectory(subjdir)
    analyzer = annif.analyzer.get_analyzer('snowball(english)')
    tfidf.load_subjects(subjects, analyzer)
    assert tmpdir.join('backends/tfidf/dictionary').exists()
    assert tmpdir.join('backends/tfidf/dictionary').size() > 0
    assert tmpdir.join('backends/tfidf/tfidf').exists()
    assert tmpdir.join('backends/tfidf/tfidf').size() > 0
    assert tmpdir.join('backends/tfidf/index').exists()
    assert tmpdir.join('backends/tfidf/index').size() > 0
