"""Configuration file handling"""


import os.path
import configparser
import tomli
import annif
import annif.util
from glob import glob
from annif.exception import ConfigurationException


logger = annif.logger


class AnnifConfigCFG:
    """Class for reading configuration in CFG/INI format"""

    def __init__(self, filename):
        self._config = configparser.ConfigParser()
        self._config.optionxform = annif.util.identity
        with open(filename, encoding='utf-8-sig') as projf:
            try:
                logger.debug(
                    f"Reading configuration file {filename} in CFG format")
                self._config.read_file(projf)
            except (configparser.DuplicateOptionError,
                    configparser.DuplicateSectionError) as err:
                raise ConfigurationException(err)

    @property
    def project_ids(self):
        return self._config.sections()

    def __getitem__(self, key):
        return self._config[key]


class AnnifConfigTOML:
    """Class for reading configuration in TOML format"""

    def __init__(self, filename):
        with open(filename, "rb") as projf:
            try:
                logger.debug(
                    f"Reading configuration file {filename} in TOML format")
                self._config = tomli.load(projf)
            except tomli.TOMLDecodeError as err:
                raise ConfigurationException(
                    f"Parsing TOML file '{filename}' failed: {err}")

    @property
    def project_ids(self):
        return self._config.keys()

    def __getitem__(self, key):
        return self._config[key]


class AnnifConfigDirectory:
    """Class for reading configuration from directory"""

    def __init__(self, directory):
        files = glob(os.path.join(directory, '*.cfg'))
        files.extend(glob(os.path.join(directory, '*.toml')))
        logger.debug(f"Reading configuration files in directory {directory}")

        self._config = dict()
        for file in files:
            source_config = parse_config(file)
            for proj_id in source_config.project_ids:
                self._check_duplicate_project_ids(proj_id, file)
                self._config[proj_id] = source_config[proj_id]

    def _check_duplicate_project_ids(self, proj_id, file):
        if proj_id in self._config:
            # Error message resembles configparser's DuplicateSection message
            raise ConfigurationException(
                f'While reading from "{file}": project ID "{proj_id}" already '
                'exists in another configuration file in the directory.')

    @property
    def project_ids(self):
        return self._config.keys()

    def __getitem__(self, key):
        return self._config[key]


def check_config(projects_config_path):
    if os.path.exists(projects_config_path):
        return projects_config_path
    else:
        logger.warning(
            'Project configuration file or directory ' +
            f'"{projects_config_path}" is missing. Please provide one. ' +
            'You can set the path to the project configuration ' +
            'using the ANNIF_PROJECTS environment ' +
            'variable or the command-line option "--projects".')
        return None


def find_config():
    for path in ('projects.cfg', 'projects.toml', 'projects.d'):
        if os.path.exists(path):
            return path

    logger.warning(
        'Could not find project configuration ' +
        '"projects.cfg", "projects.toml" or "projects.d". ' +
        'You can set the path to the project configuration ' +
        'using the ANNIF_PROJECTS environment ' +
        'variable or the command-line option "--projects".')
    return None


def parse_config(projects_config_path):
    if projects_config_path:
        projects_config_path = check_config(projects_config_path)
    else:
        projects_config_path = find_config()

    if not projects_config_path:  # not found
        return None

    if os.path.isdir(projects_config_path):
        return AnnifConfigDirectory(projects_config_path)
    elif projects_config_path.endswith('.toml'):  # TOML format
        return AnnifConfigTOML(projects_config_path)
    else:  # classic CFG/INI style format
        return AnnifConfigCFG(projects_config_path)
