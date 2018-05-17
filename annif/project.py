"""Project management functionality for Annif"""

import collections
import configparser
import os.path
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import current_app
import annif
import annif.analyzer
import annif.corpus
import annif.hit
import annif.backend
import annif.util
from annif import logger


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    # defaults for unitialized instances
    _analyzer = None
    _subjects = None
    _vectorizer = None
    initialized = False

    def __init__(self, project_id, config, datadir):
        self.project_id = project_id
        self.name = config['name']
        self.language = config['language']
        self.analyzer_spec = config.get('analyzer', None)
        self._datadir = os.path.join(datadir, 'projects', self.project_id)
        self.backends = self._initialize_backends(config)

    def _get_datadir(self):
        """return the path of the directory where this project can store its
        data files"""
        if not os.path.exists(self._datadir):
            os.makedirs(self._datadir)
        return self._datadir

    def _initialize_backends(self, config):
        backends = []
        for backenddef in config['backends'].split(','):
            bedefs = backenddef.strip().split(':')
            backend_id = bedefs[0]
            if len(bedefs) > 1:
                weight = float(bedefs[1])
            else:
                weight = 1.0
            backend_type = annif.backend.get_backend(backend_id)
            backend = backend_type(
                backend_id,
                params=config,
                datadir=self._datadir)
            backends.append((backend, weight))
        return backends

    def initialize(self):
        """initialize this project and all backends so that they are ready to
        analyze"""
        logger.debug("Initializing project '%s'", self.project_id)
        analyzer = self.analyzer
        logger.debug("Project '%s': initialized analyzer: %s",
                     self.project_id,
                     str(analyzer))
        subjects = self.subjects
        logger.debug("Project '%s': initialized subjects: %s",
                     self.project_id,
                     str(subjects))
        vectorizer = self.vectorizer
        logger.debug("Project '%s': initialized vectorizer: %s",
                     self.project_id,
                     str(vectorizer))

        logger.debug("Project '%s': initializing backends", self.project_id)
        for backend, weight in self.backends:
            backend.initialize()

        self.initialized = True

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
                'Got %d hits from backend %s',
                len(hits), backend.backend_id)
            for hit in hits:
                hits_by_uri[hit.uri].append((hit.score * weight, hit))
        return hits_by_uri

    def _merge_hits(self, hits_by_uri):
        merged_hits = []
        for score_hits in hits_by_uri.values():
            totalweight = sum((be[1] for be in self.backends))
            total = sum([sh[0] for sh in score_hits]) / totalweight
            hit = score_hits[0][1]._replace(score=total)
            merged_hits.append(hit)
        return merged_hits

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = annif.analyzer.get_analyzer(self.analyzer_spec)
        return self._analyzer

    @property
    def subjects(self):
        if self._subjects is None:
            path = os.path.join(self._get_datadir(), 'subjects')
            if os.path.exists(path):
                logger.debug('loading subjects from %s', path)
                self._subjects = annif.corpus.SubjectIndex.load(path)
            else:
                logger.warning("subject file '%s' not found", path)
        return self._subjects

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            path = os.path.join(self._get_datadir(), 'vectorizer')
            if os.path.exists(path):
                logger.debug('loading vectorizer from %s', path)
                self._vectorizer = joblib.load(path)
            else:
                logger.warning("vectorizer file '%s' not found", path)
        return self._vectorizer

    def analyze(self, text, backend_params=None):
        """Analyze the given text by passing it to backends and joining the
        results. Returns a list of AnalysisHit objects ordered by decreasing
        score."""

        logger.debug('Analyzing text "%s..." (len=%d)',
                     text[:20], len(text))
        hits_by_uri = self._analyze_with_backends(text, backend_params)
        merged_hits = self._merge_hits(hits_by_uri)
        logger.debug('%d hits after merging', len(merged_hits))
        return merged_hits

    def _create_subject_index(self, subjects):
        if True not in [be[0].needs_subject_index for be in self.backends]:
            logger.debug(
                'not creating subject index: not needed by any backend')
            return
        logger.info('creating subject index')
        if isinstance(subjects, annif.corpus.SubjectIndex):
            self._subjects = subjects
        else:
            self._subjects = annif.corpus.SubjectIndex(subjects)
        annif.util.atomic_save(self._subjects, self._get_datadir(), 'subjects')

    def _create_vectorizer(self, subjects):
        if True not in [
                be[0].needs_subject_vectorizer for be in self.backends]:
            logger.debug('not creating vectorizer: not needed by any backend')
            return
        logger.info('creating vectorizer')
        self._vectorizer = TfidfVectorizer(
            tokenizer=self.analyzer.tokenize_words)
        self._vectorizer.fit((subj.text for subj in subjects))
        annif.util.atomic_save(
            self._vectorizer,
            self._get_datadir(),
            'vectorizer',
            method=joblib.dump)

    def load_subjects(self, subjects):
        self._create_subject_index(subjects)
        self._create_vectorizer(subjects)

        for backend, weight in self.backends:
            logger.debug(
                'Loading subjects for backend %s',
                backend.backend_id)
            backend.load_subjects(subjects, project=self)

    def load_vocabulary(self, subjects):
        self._create_subject_index(subjects)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'name': self.name,
                'language': self.language,
                'backends': [{'backend_id': be[0].backend_id,
                              'weight': be[1]} for be in self.backends]
                }


def _create_projects(projects_file, datadir, init_projects):
    config = configparser.ConfigParser()
    with open(projects_file) as projf:
        config.read_file(projf)

    # create AnnifProject objects from the configuration file
    projects = collections.OrderedDict()
    for project_id in config.sections():
        projects[project_id] = AnnifProject(project_id,
                                            config[project_id],
                                            datadir)
        if init_projects:
            projects[project_id].initialize()
    return projects


def initialize_projects(app):
    projects_file = app.config['PROJECTS_FILE']
    datadir = app.config['DATADIR']
    init_projects = app.config['INITIALIZE_PROJECTS']
    app.annif_projects = _create_projects(
        projects_file, datadir, init_projects)


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
