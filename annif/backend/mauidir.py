"""Maui backend that cheats by looking at a preprocessed directory of .txt
files with corresponding .maui files containing subjects determined by Maui"""


import glob
import hashlib
import os.path
import re
from annif.hit import AnalysisHit
from . import backend


class MauiDirBackend(backend.AnnifBackend):
    name = "mauidir"

    _doc_subjects = None

    @classmethod
    def _hash(cls, text):
        return hashlib.sha1(text.encode('utf-8')).hexdigest()

    def initialize(self, dir):
        if self._doc_subjects is not None:
            return
        self._doc_subjects = {}
        self.info('loading subjects from {}'.format(dir))
        for filename in glob.glob(os.path.join(dir, '*.txt')):
            self.debug('reading {}'.format(filename))
            mauifilename = re.sub(r'\.txt$', '.maui', filename)
            with open(filename) as txtfile:
                text = txtfile.read()
                hash = self._hash(text)
            subjects = []
            with open(mauifilename) as mauifile:
                for line in mauifile:
                    label, score = line.split("\t")
                    score = float(score)
                    subjects.append((label, score))
            self._doc_subjects[hash] = subjects
        self.info('initialized {} doc subjects'.format(
            len(self._doc_subjects)))

    def _analyze(self, text, project, params):
        self.initialize(params['directory'])
        results = []
        for label, score in self._doc_subjects[self._hash(text)]:
            subject = project.subjects.by_label(label)
            if subject is None:
                continue
            results.append(AnalysisHit(uri=subject[0],
                                       label=subject[1],
                                       score=score))
        return results
