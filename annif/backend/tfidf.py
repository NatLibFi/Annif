"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import collections
import os
import os.path
import tempfile
import gensim.corpora
import gensim.models
import gensim.similarities
import annif.analyzer
import annif.corpus
from annif.hit import AnalysisHit
from . import backend


class VectorCorpus:
    """A class that wraps a subject corpus so it can be iterated as lists of
    vectors, by using a dictionary to map words to integers."""

    def __init__(self, corpus, dictionary, analyzer):
        self.corpus = corpus
        self.dictionary = dictionary
        self.analyzer = analyzer

    def __iter__(self):
        """Iterate through the subject directory, yielding vectors that are
        derived from subjects using the given analyzer and dictionary."""

        for subject in self.corpus:
            yield self.dictionary.doc2bow(self.analyzer.tokenize_words(subject.text))


class SubjectIndex:
    """A class that remembers the associations between integers subject IDs
    and their URIs and labels."""

    def __init__(self, corpus):
        """Initialize the subject index from a subject corpus."""
        self._uris = []
        self._labels = []
        for subject_id, subject in enumerate(corpus):
            self._uris.append(subject.uri)
            self._labels.append(subject.label)

    def __getitem__(self, subject_id):
        return (self._uris[subject_id], self._labels[subject_id])

    def save(self, path):
        """Save this subject index into a file."""

        with open(path, 'w') as subjfile:
            for subject_id in range(len(self._uris)):
                line = "<{}>\t{}".format(
                    self._uris[subject_id], self._labels[subject_id])
                print(line, file=subjfile)

    @classmethod
    def load(cls, path):
        """Load a subject index from a file and return it."""

        def file_as_corpus(path):
            with open(path) as subjfile:
                for line in subjfile:
                    uri, label = line.strip().split(None, 1)
                    yield annif.corpus.Subject(uri, label, None)

        return cls(file_as_corpus(path))


class TFIDFBackend(backend.AnnifBackend):
    name = "tfidf"

    # top K subjects per chunk to consider
    MAX_CHUNK_SUBJECTS = 100

    # defaults for uninitialized instances
    _analyzer = None
    _dictionary = None
    _tfidf = None
    _index = None

    def _atomic_save(self, obj, dirname, filename):
        tempfd, tempfilename = tempfile.mkstemp(prefix=filename, dir=dirname)
        os.close(tempfd)
        obj.save(tempfilename)
        os.rename(tempfilename, os.path.join(dirname, filename))

    def _initialize_subjects(self):
        if self._subjects is None:
            path = os.path.join(self._get_datadir(), 'subjects')
            self._subjects = SubjectIndex.load(path)

    def _initialize_analyzer(self):
        if self._analyzer is None:
            self._analyzer = annif.analyzer.get_analyzer(
                self.params['analyzer'])

    def _initialize_dictionary(self):
        if self._dictionary is None:
            path = os.path.join(self._get_datadir(), 'dictionary')
            self._dictionary = gensim.corpora.Dictionary.load(path)

    def _initialize_tfidf(self):
        if self._tfidf is None:
            path = os.path.join(self._get_datadir(), 'tfidf')
            self._tfidf = gensim.models.TfidfModel.load(path)

    def _initialize_index(self):
        if self._index is None:
            path = os.path.join(self._get_datadir(), 'index')
            self._index = gensim.similarities.SparseMatrixSimilarity.load(path)

    def initialize(self):
        self._initialize_subjects()
        self._initialize_analyzer()
        self._initialize_dictionary()
        self._initialize_tfidf()
        self._initialize_index()

    def load_subjects(self, subjects):
        self._subjects = SubjectIndex(subjects)
        self._atomic_save(self._subjects, self._get_datadir(), 'subjects')
        self._initialize_analyzer()
        self._dictionary = gensim.corpora.Dictionary(
            (self._analyzer.tokenize_words(subject.text) for subject in subjects))
        self._atomic_save(self._dictionary, self._get_datadir(), 'dictionary')
        veccorpus = VectorCorpus(subjects, self._dictionary, self._analyzer)
        self._tfidf = gensim.models.TfidfModel(veccorpus)
        self._atomic_save(self._tfidf, self._get_datadir(), 'tfidf')
        self._index = gensim.similarities.SparseMatrixSimilarity(
            self._tfidf[veccorpus], num_features=len(self._dictionary))
        self._atomic_save(self._index, self._get_datadir(), 'index')

    def _analyze_chunks(self, chunks):
        results = []
        for docsim in self._index[chunks]:
            sims = sorted(
                enumerate(docsim),
                key=lambda item: item[1],
                reverse=True)
            results.append(sims[:self.MAX_CHUNK_SUBJECTS])
        return results

    def _merge_chunk_results(self, chunk_results):
        subject_scores = collections.defaultdict(float)
        for result in chunk_results:
            for subject_id, score in result:
                subject_scores[subject_id] += score
        best_subjects = sorted([(score,
                                 subject_id) for subject_id,
                                score in subject_scores.items()],
                               reverse=True)
        limit = int(self.params['limit'])
        results = []
        for score, subject_id in best_subjects[:limit]:
            subject = self._subjects[subject_id]
            results.append(
                AnalysisHit(
                    subject[0],
                    subject[1],
                    score /
                    len(chunk_results)))
        return results

    def analyze(self, text):
        self.initialize()
        sentences = self._analyzer.tokenize_sentences(text)
        chunksize = int(self.params['chunksize'])
        chunks = []  # chunks represented as TF-IDF normalized vectors
        for i in range(0, len(sentences), chunksize):
            chunktext = ' '.join(sentences[i:i + chunksize])
            chunkbow = self._dictionary.doc2bow(
                self._analyzer.tokenize_words(chunktext))
            chunks.append(self._tfidf[chunkbow])
        chunk_results = self._analyze_chunks(chunks)
        return self._merge_chunk_results(chunk_results)
