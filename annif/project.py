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
import annif.vocab
from annif.exception import AnnifException, ConfigurationException, \
    NotInitializedException

logger = annif.logger


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    # defaults for uninitialized instances
    _analyzer = None
    _backend = None
    _vocab = None
    _vectorizer = None
    initialized = False

    def __init__(self, project_id, config, datadir):
        self.project_id = project_id
        self.name = config['name']
        self.language = config['language']
        self.analyzer_spec = config.get('analyzer', None)
        self.vocab_id = config.get('vocab', None)
        self._base_datadir = datadir
        self._datadir = os.path.join(datadir, 'projects', self.project_id)
        self.config = config

    def _get_datadir(self):
        """return the path of the directory where this project can store its
        data files"""
        if not os.path.exists(self._datadir):
            os.makedirs(self._datadir)
        return self._datadir

    def _initialize_analyzer(self):
        analyzer = self.analyzer
        logger.debug("Project '%s': initialized analyzer: %s",
                     self.project_id,
                     str(analyzer))

    def _initialize_subjects(self):
        try:
            subjects = self.subjects
            logger.debug("Project '%s': initialized subjects: %s",
                         self.project_id,
                         str(subjects))
        except AnnifException as err:
            logger.warning(err.format_message())

    def _initialize_vectorizer(self):
        try:
            vectorizer = self.vectorizer
            logger.debug("Project '%s': initialized vectorizer: %s",
                         self.project_id,
                         str(vectorizer))
        except AnnifException as err:
            logger.warning(err.format_message())

    def _initialize_backend(self):
        logger.debug("Project '%s': initializing backend", self.project_id)
        if not self.backend:
            logger.debug("Cannot initialize backend: does not exist")
            return
        try:
            self.backend.initialize()
        except AnnifException as err:
            logger.warning(err.format_message())

    def initialize(self):
        """initialize this project and its backend so that they are ready to
        analyze"""
        logger.debug("Initializing project '%s'", self.project_id)

        self._initialize_analyzer()
        self._initialize_subjects()
        self._initialize_vectorizer()
        self._initialize_backend()

        self.initialized = True

    def _analyze_with_backend(self, text, backend_params):
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        hits = self.backend.analyze(text, project=self, params=beparams)
        logger.debug(
            'Got %d hits from backend %s',
            len(hits), self.backend.backend_id)
        return hits

    @property
    def analyzer(self):
        if self._analyzer is None and self.analyzer_spec:
            self._analyzer = annif.analyzer.get_analyzer(self.analyzer_spec)
        return self._analyzer

    @property
    def backend(self):
        if self._backend is None:
            backend_id = self.config['backend']
            try:
                backend_class = annif.backend.get_backend(backend_id)
                self._backend = backend_class(
                    backend_id, params=self.config, datadir=self._datadir)
            except ValueError:
                logger.debug("Could not create backend %s", backend_id)
        return self._backend

    @property
    def vocab(self):
        if self._vocab is None:
            if self.vocab_id is None:
                raise ConfigurationException("vocab setting is missing",
                                             project_id=self.project_id)
            self._vocab = annif.vocab.AnnifVocabulary(self.vocab_id,
                                                      self._base_datadir)
        return self._vocab

    @property
    def subjects(self):
        return self.vocab.subjects

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            path = os.path.join(self._get_datadir(), 'vectorizer')
            if os.path.exists(path):
                logger.debug('loading vectorizer from %s', path)
                self._vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    project_id=self.project_id)
        return self._vectorizer

    def analyze(self, text, backend_params=None):
        """Analyze the given text by passing it to the backend. Returns a
        list of AnalysisHit objects ordered by decreasing score."""

        logger.debug('Analyzing text "%s..." (len=%d)',
                     text[:20], len(text))
        hits = self._analyze_with_backend(text, backend_params)
        logger.debug('%d hits from backend', len(hits))
        return hits

    def _create_vectorizer(self, subjectcorpus):
        if not self.backend.needs_subject_vectorizer:
            logger.debug('not creating vectorizer: not needed by backend')
            return
        logger.info('creating vectorizer')
        self._vectorizer = TfidfVectorizer(
            tokenizer=self.analyzer.tokenize_words)
        self._vectorizer.fit((subj.text for subj in subjectcorpus.subjects))
        annif.util.atomic_save(
            self._vectorizer,
            self._get_datadir(),
            'vectorizer',
            method=joblib.dump)

    def load_documents(self, corpus):
        """load training documents from a metadata source"""

        corpus.set_subject_index(self.subjects)
        self._create_vectorizer(corpus)
        self.backend.load_corpus(corpus, project=self)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'name': self.name,
                'language': self.language,
                'backend': {'backend_id': self.config['backend']}
                }


def _create_projects(projects_file, datadir, init_projects):
    if not os.path.exists(projects_file):
        logger.warning("Project configuration file '%s' is missing. " +
                       'Please provide one.', projects_file)
        logger.warning('You can set the path to the project configuration ' +
                       'file using the ANNIF_PROJECTS environment variable.')
        return {}

    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
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
