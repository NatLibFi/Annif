"""Annif backend using the fastText classifier"""

import collections
import os.path
import annif.util
from annif.hit import AnalysisHit, ListAnalysisResult
from annif.exception import NotInitializedException
import fastText
from . import backend


class FastTextBackend(backend.AnnifBackend):
    """fastText backend for Annif"""

    name = "fasttext"
    needs_subject_index = True

    FASTTEXT_PARAMS = {
        'lr': float,
        'lrUpdateRate': int,
        'dim': int,
        'ws': int,
        'epoch': int,
        'minCount': int,
        'neg': int,
        'wordNgrams': int,
        'loss': str,
        'bucket': int,
        'minn': int,
        'maxn': int,
        'thread': int,
        't': float
    }

    MODEL_FILE = 'fasttext-model'
    TRAIN_FILE = 'fasttext-train.txt'

    # defaults for uninitialized instances
    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self._get_datadir(), self.MODEL_FILE)
            self.debug('loading fastText model from {}'.format(path))
            if os.path.exists(path):
                self._model = fastText.load_model(path)
                self.debug('loaded model {}'.format(str(self._model)))
                self.debug('dim: {}'.format(self._model.get_dimension()))
            else:
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)

    @classmethod
    def _id_to_label(cls, subject_id):
        return "__label__{:d}".format(subject_id)

    @classmethod
    def _label_to_subject(cls, project, label):
        labelnum = label.replace('__label__', '')
        subject_id = int(labelnum)
        return project.subjects[subject_id]

    def _write_train_file(self, doc_subjects, filename):
        with open(filename, 'w') as trainfile:
            for doc, subject_ids in doc_subjects.items():
                labels = [self._id_to_label(sid) for sid in subject_ids
                          if sid is not None]
                if labels:
                    print(' '.join(labels), doc, file=trainfile)
                else:
                    self.warning('no labels for document "{}"'.format(doc))

    @classmethod
    def _normalize_text(cls, project, text):
        return ' '.join(project.analyzer.tokenize_words(text))

    def _create_train_file(self, corpus, project):
        self.info('creating fastText training file')

        doc_subjects = collections.defaultdict(set)

        for doc in corpus.documents:
            text = self._normalize_text(project, doc.text)
            if text == '':
                continue
            doc_subjects[text] = [project.subjects.by_uri(uri)
                                  for uri in doc.uris]

        annif.util.atomic_save(doc_subjects,
                               self._get_datadir(),
                               self.TRAIN_FILE,
                               method=self._write_train_file)

    def _create_model(self):
        self.info('creating fastText model')
        trainpath = os.path.join(self._get_datadir(), self.TRAIN_FILE)
        modelpath = os.path.join(self._get_datadir(), self.MODEL_FILE)
        params = {param: self.FASTTEXT_PARAMS[param](val)
                  for param, val in self.params.items()
                  if param in self.FASTTEXT_PARAMS}
        self._model = fastText.train_supervised(trainpath, **params)
        self._model.save_model(modelpath)

    def load_corpus(self, corpus, project):
        self._create_train_file(corpus, project)
        self._create_model()

    def _analyze_chunks(self, chunktexts, project):
        limit = int(self.params['limit'])
        chunklabels, chunkscores = self._model.predict(chunktexts, limit)
        label_scores = collections.defaultdict(float)
        for labels, scores in zip(chunklabels, chunkscores):
            for label, score in zip(labels, scores):
                label_scores[label] += score
        best_labels = sorted([(score, label)
                              for label, score in label_scores.items()],
                             reverse=True)

        results = []
        for score, label in best_labels[:limit]:
            subject = self._label_to_subject(project, label)
            results.append(AnalysisHit(
                uri=subject[0],
                label=subject[1],
                score=score / len(chunktexts)))
        return ListAnalysisResult(results, project.subjects)

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        sentences = project.analyzer.tokenize_sentences(text)
        self.debug('Found {} sentences'.format(len(sentences)))
        chunksize = int(params['chunksize'])
        chunktexts = []
        for i in range(0, len(sentences), chunksize):
            chunktext = ' '.join(sentences[i:i + chunksize])
            normalized = self._normalize_text(project, chunktext)
            if normalized != '':
                chunktexts.append(normalized)
        self.debug('Split sentences into {} chunks'.format(len(chunktexts)))

        return self._analyze_chunks(chunktexts, project)
