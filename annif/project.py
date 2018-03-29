"""Project management functionality for Annif"""

import collections
import configparser
import os.path
import gensim.corpora
import gensim.models
from flask import current_app
import annif
import annif.analyzer
import annif.hit
import annif.backend
import annif.util
from annif import logger


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    # defaults for unitialized instances
    _analyzer = None
    _subjects = None
    _dictionary = None
    _tfidf = None

    def __init__(self, project_id, config, datadir, all_backends):
        self.project_id = project_id
        self.language = config['language']
        self.analyzer_spec = config['analyzer']
        self.backends = self._initialize_backends(config['backends'],
                                                  all_backends)
        self._datadir = os.path.join(datadir, 'projects', self.project_id)

    def _get_datadir(self):
        """return the path of the directory where this project can store its
        data files"""
        if not os.path.exists(self._datadir):
            os.makedirs(self._datadir)
        return self._datadir

    def _initialize_backends(self, backends_configuration, all_backends):
        backends = []
        for backenddef in backends_configuration.split(','):
            bedefs = backenddef.strip().split(':')
            backend_id = bedefs[0]
            if len(bedefs) > 1:
                weight = float(bedefs[1])
            else:
                weight = 1.0
            backend = all_backends[backend_id]
            backends.append((backend, weight))
        return backends

    def _analyze_with_backends(self, text, backend_params):
        if backend_params is None:
            backend_params = {}
        hits_by_uri = collections.defaultdict(list)
        for backend, weight in self.backends:
            beparams = backend_params.get(backend.backend_id, {})
            hits = [
                hit for hit in backend.analyze(
                    text,
                    project=self,
                    params=beparams) if hit.score > 0.0]
            logger.debug(
                'Got {} hits from backend {}'.format(
                    len(hits), backend.backend_id))
            for hit in hits:
                hits_by_uri[hit.uri].append((hit.score * weight, hit))
        return hits_by_uri

    @classmethod
    def _merge_hits(cls, hits_by_uri):
        merged_hits = []
        for score_hits in hits_by_uri.values():
            total = sum([sh[0] for sh in score_hits])
            hit = annif.hit.AnalysisHit(
                score_hits[0][1].uri, score_hits[0][1].label, total)
            merged_hits.append(hit)
        return merged_hits

    @classmethod
    def _filter_hits(cls, hits, limit, threshold):
        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = hits[:limit]
        logger.debug(
            '{} hits after applying limit {}'.format(
                len(hits), limit))
        hits = [hit for hit in hits if hit.score >= threshold]
        logger.debug(
            '{} hits after applying threshold {}'.format(
                len(hits), threshold))
        return hits

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = annif.analyzer.get_analyzer(self.analyzer_spec)
        return self._analyzer

    @property
    def subjects(self):
        if self._subjects is None:
            path = os.path.join(self._get_datadir(), 'subjects')
            logger.debug('loading subjects from {}'.format(path))
            self._subjects = annif.corpus.SubjectIndex.load(path)
        return self._subjects

    @property
    def dictionary(self):
        if self._dictionary is None:
            path = os.path.join(self._get_datadir(), 'dictionary')
            logger.debug('loading dictionary from {}'.format(path))
            self._dictionary = gensim.corpora.Dictionary.load(path)
        return self._dictionary

    @property
    def tfidf(self):
        if self._tfidf is None:
            path = os.path.join(self._get_datadir(), 'tfidf')
            logger.debug('loading TF-IDF model from {}'.format(path))
            self._tfidf = gensim.models.TfidfModel.load(path)
        return self._tfidf

    def analyze(self, text, limit=10, threshold=0.0, backend_params=None):
        """Analyze the given text by passing it to backends and joining the
        results. Returns a list of AnalysisHit objects ordered by decreasing
        score. The limit parameter defines the maximum number of hits to
        return. Only hits whose score is over the threshold are returned."""

        logger.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        hits_by_uri = self._analyze_with_backends(text, backend_params)
        merged_hits = self._merge_hits(hits_by_uri)
        logger.debug('{} hits after merging'.format(len(merged_hits)))
        return self._filter_hits(merged_hits, limit, threshold)

    def _create_subject_index(self, subjects):
        logger.info('creating subject index')
        self._subjects = annif.corpus.SubjectIndex(subjects)
        annif.util.atomic_save(self._subjects, self._get_datadir(), 'subjects')

    def _create_dictionary(self, subjects):
        logger.info('creating dictionary')
        self._dictionary = gensim.corpora.Dictionary(
            (self.analyzer.tokenize_words(subject.text)
             for subject in subjects))
        annif.util.atomic_save(
            self._dictionary,
            self._get_datadir(),
            'dictionary')

    def _create_tfidf(self, subjects):
        veccorpus = annif.corpus.VectorCorpus(subjects,
                                              self.dictionary,
                                              self.analyzer)
        logger.info('creating TF-IDF model')
        self._tfidf = gensim.models.TfidfModel(veccorpus)
        annif.util.atomic_save(self._tfidf, self._get_datadir(), 'tfidf')

    def load_subjects(self, subjects):
        self._create_subject_index(subjects)
        self._create_dictionary(subjects)
        self._create_tfidf(subjects)

        for backend, weight in self.backends:
            logger.debug(
                'Loading subjects for backend {}'.format(
                    backend.backend_id))
            backend.load_subjects(subjects, project=self)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'language': self.language,
                'backends': [{'backend_id': be[0].backend_id,
                              'weight': be[1]} for be in self.backends]
                }


def _create_projects(projects_file, datadir, backends):
    config = configparser.ConfigParser()
    with open(projects_file) as projf:
        config.read_file(projf)

    # create AnnifProject objects from the configuration file
    projects = {}
    for project_id in config.sections():
        projects[project_id] = AnnifProject(project_id,
                                            config[project_id],
                                            datadir,
                                            backends)
    return projects


def init_projects(app, backends):
    projects_file = app.config['PROJECTS_FILE']
    datadir = app.config['DATADIR']
    app.annif_projects = _create_projects(projects_file, datadir, backends)


def get_projects():
    """return the available projects as a dict of project_id -> AnnifProject"""
    return current_app.annif_projects


def get_project(project_id):
    """return the definition of a single Project by project_id"""
    projects = get_projects()
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
