"""Project management functionality for Annif"""

import collections
import configparser
import enum
import os.path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import current_app
import annif
import annif.analyzer
import annif.corpus
import annif.suggestion
import annif.backend
import annif.util
import annif.vocab
from annif.datadir import DatadirMixin
from annif.exception import AnnifException, ConfigurationException, \
    NotInitializedException, NotSupportedException

logger = annif.logger


class Access(enum.IntEnum):
    """Enumeration of access levels for projects"""
    private = 1
    hidden = 2
    public = 3


class AnnifProject(DatadirMixin):
    """Class representing the configuration of a single Annif project."""

    # defaults for uninitialized instances
    _analyzer = None
    _backend = None
    _vocab = None
    _vectorizer = None
    initialized = False

    # default values for configuration settings
    DEFAULT_ACCESS = 'public'

    def __init__(self, project_id, config, datadir):
        DatadirMixin.__init__(self, datadir, 'projects', project_id)
        self.project_id = project_id
        self.name = config['name']
        self.language = config['language']
        self.analyzer_spec = config.get('analyzer', None)
        self.vocab_id = config.get('vocab', None)
        self.config = config
        self._base_datadir = datadir
        self._init_access()

    def _init_access(self):
        access = self.config.get('access', self.DEFAULT_ACCESS)
        try:
            self.access = getattr(Access, access)
        except AttributeError:
            raise ConfigurationException(
                "'{}' is not a valid access setting".format(access),
                project_id=self.project_id)

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
        be used"""

        logger.debug("Initializing project '%s'", self.project_id)

        self._initialize_analyzer()
        self._initialize_subjects()
        self._initialize_vectorizer()
        self._initialize_backend()

        self.initialized = True

    def _suggest_with_backend(self, text, backend_params):
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        hits = self.backend.suggest(text, project=self, params=beparams)
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
                    backend_id, params=self.config, datadir=self.datadir)
            except ValueError:
                logger.warning(
                    "Could not create backend %s, "
                    "make sure you've installed optional dependencies",
                    backend_id)
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
            path = os.path.join(self.datadir, 'vectorizer')
            if os.path.exists(path):
                logger.debug('loading vectorizer from %s', path)
                self._vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    project_id=self.project_id)
        return self._vectorizer

    def suggest(self, text, backend_params=None):
        """Suggest subjects the given text by passing it to the backend. Returns a
        list of SubjectSuggestion objects ordered by decreasing score."""

        logger.debug('Suggesting subjects for text "%s..." (len=%d)',
                     text[:20], len(text))
        hits = self._suggest_with_backend(text, backend_params)
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
            self.datadir,
            'vectorizer',
            method=joblib.dump)

    def train(self, corpus):
        """train the project using documents from a metadata source"""

        corpus.set_subject_index(self.subjects)
        self._create_vectorizer(corpus)
        self.backend.train(corpus, project=self)

    def learn(self, corpus):
        """further train the project using documents from a metadata source"""

        corpus.set_subject_index(self.subjects)
        if isinstance(
                self.backend,
                annif.backend.backend.AnnifLearningBackend):
            self.backend.learn(corpus, project=self)
        else:
            raise NotSupportedException("Learning not supported by backend",
                                        project_id=self.project_id)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'name': self.name,
                'language': self.language,
                'backend': {'backend_id': self.config['backend']}
                }


def _create_projects(projects_file, datadir, init_projects):
    if not os.path.exists(projects_file):
        logger.warning(
            'Project configuration file "%s" is missing. Please provide one.' +
            ' You can set the path to the project configuration file using ' +
            'the ANNIF_PROJECTS environment variable or the command-line ' +
            'option "--projects".', projects_file)
        return {}

    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    with open(projects_file, encoding='utf-8') as projf:
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


def get_projects(min_access=Access.private):
    """Return the available projects as a dict of project_id ->
    AnnifProject. The min_access parameter may be used to set the minimum
    access level required for the returned projects."""

    if not hasattr(current_app, 'annif_projects'):
        initialize_projects(current_app)

    projects = [(project_id, project)
                for project_id, project in current_app.annif_projects.items()
                if project.access >= min_access]
    return collections.OrderedDict(projects)


def get_project(project_id, min_access=Access.private):
    """return the definition of a single Project by project_id"""
    projects = get_projects(min_access)
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
