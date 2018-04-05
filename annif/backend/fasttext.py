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
            self.debug("Subject:" + str(subj))
            self.debug("Subject id:" + str(subject_id))
            for line in subj.text.splitlines():
                doc_subjects[line].add(subject_id)
                self.debug(line)

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
        normalized_text = self._normalize_text(project, text)
        if normalized_text == '':
            return []
        ft_results = self._model.predict_proba([text], self.params['limit'])
        results = []
        for label, score in ft_results[0]:
            subject = self._label_to_subject(project, label)
            results.append(AnalysisHit(
                uri=subject[0],
                label=subject[1],
                score=score))
        return results
