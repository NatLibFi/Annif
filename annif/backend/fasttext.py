"""Annif backend using the fastText classifier"""

import collections
import os.path
import annif.util
from annif.hit import AnalysisHit
import fasttext
from . import backend


class FastTextBackend(backend.AnnifBackend):
    name = "fasttext"
    needs_subject_index = True

    FASTTEXT_PARAMS = (
        'lr',
        'lr_update_rate',
        'dim',
        'ws',
        'epoch',
        'min_count',
        'neg',
        'word_ngrams',
        'loss',
        'bucket',
        'minn',
        'maxn',
        'thread',
        't'
    )

    # defaults for uninitialized instances
    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self._get_datadir(), 'model.bin')
            self.debug('loading fastText model from {}'.format(path))
            self._model = fasttext.load_model(path)
            self.debug('loaded model {}'.format(str(self._model)))
            self.debug('dim: {}'.format(self._model.dim))
            self.debug('epoch: {}'.format(self._model.epoch))
            self.debug('loss_name: {}'.format(self._model.loss_name))

    @classmethod
    def _id_to_label(cls, subject_id):
        return "__label__{:d}".format(subject_id)

    @classmethod
    def _label_to_subject(cls, project, label):
        subject_id = int(label.replace('__label__', ''))
        return project.subjects[subject_id]

    @classmethod
    def _write_train_file(cls, doc_subjects, filename):
        with open(filename, 'w') as trainfile:
            for doc, subject_ids in doc_subjects.items():
                labels = [cls._id_to_label(sid) for sid in subject_ids]
                print(' '.join(labels), doc, file=trainfile)

    @classmethod
    def _normalize_text(cls, project, text):
        return ' '.join(project.analyzer.tokenize_words(text))

    def load_subjects(self, subjects, project):
        self.info('creating fastText training file')

        doc_subjects = collections.defaultdict(set)
        for subject_id, subj in enumerate(subjects):
            for line in subj.text.splitlines():
                doc_subjects[line].add(subject_id)

        doc_subjects_normalized = {}
        for doc, subjects in doc_subjects.items():
            text = self._normalize_text(project, doc)
            if text != '':
                doc_subjects_normalized[text] = subjects

        annif.util.atomic_save(doc_subjects_normalized,
                               self._get_datadir(),
                               'train.txt',
                               method=self._write_train_file)

        self.info('creating fastText model')
        trainpath = os.path.join(self._get_datadir(), 'train.txt')
        modelpath = os.path.join(self._get_datadir(), 'model')
        params = {param: val for param, val in self.params.items()
                  if param in self.FASTTEXT_PARAMS}
        self._model = fasttext.supervised(trainpath, modelpath, **params)

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
            chunktexts.append(self._normalize_text(project, chunktext))
        self.debug('Split sentences into {} chunks'.format(len(chunktexts)))
        limit = int(self.params['limit'])
        ft_results = self._model.predict_proba(chunktexts, limit)

        label_scores = collections.defaultdict(float)

        for label, score in ft_results[0]:
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
        return results
