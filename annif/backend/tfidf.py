"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import gensim.similarities
from gensim.matutils import Sparse2Corpus
import annif.util
from annif.hit import AnalysisHit
from . import backend


class TFIDFBackend(backend.AnnifBackend):
    name = "tfidf"

    # defaults for uninitialized instances
    _index = None

    def initialize(self):
        if self._index is None:
            path = os.path.join(self._get_datadir(), 'index')
            self.debug('loading similarity index from {}'.format(path))
            self._index = gensim.similarities.SparseMatrixSimilarity.load(path)

    def load_subjects(self, subjects, project):
        self.info('creating similarity index')
        veccorpus = project.vectorizer.transform(
            (subj.text for subj in subjects))
        gscorpus = Sparse2Corpus(veccorpus, documents_columns=False)
        self._index = gensim.similarities.SparseMatrixSimilarity(
            gscorpus, num_features=len(project.vectorizer.vocabulary_))
        annif.util.atomic_save(self._index, self._get_datadir(), 'index')

    def _analyze_vector(self, vector, project):
        docsim = self._index[vector]
        sims = sorted(
            enumerate(docsim),
            key=lambda item: item[1],
            reverse=True)
        results = []
        limit = int(self.params['limit'])
        for subject_id, score in sims[:limit]:
            if score <= 0.0:
                continue
            subject = project.subjects[subject_id]
            results.append(
                AnalysisHit(
                    uri=subject[0],
                    label=subject[1],
                    score=score))
        return results

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        vectors = project.vectorizer.transform([text])
        return self._analyze_vector(vectors[0], project)
