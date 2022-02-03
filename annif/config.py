"""Configuration file handling"""


import os.path
import configparser
import tomli
import annif
import annif.util
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


def check_config(projects_file):
    if os.path.exists(projects_file):
        return projects_file
    else:
        logger.warning(
            f'Project configuration file "{projects_file}" is ' +
            'missing. Please provide one. ' +
            'You can set the path to the project configuration ' +
            'file using the ANNIF_PROJECTS environment ' +
            'variable or the command-line option "--projects".')
        return None


def find_config():
    for filename in ('projects.cfg', 'projects.toml'):
        if os.path.exists(filename):
            return filename

    logger.warning(
        'Could not find project configuration file ' +
        '"projects.cfg" or "projects.toml". ' +
        'You can set the path to the project configuration ' +
        'file using the ANNIF_PROJECTS environment ' +
        'variable or the command-line option "--projects".')
    return None


def parse_config(projects_file):
    if projects_file:
        filename = check_config(projects_file)
    else:
        filename = find_config()

    if not filename:  # not found
        return None

    if filename.endswith('.toml'):  # TOML format
        return AnnifConfigTOML(filename)
    else:  # classic CFG/INI style format
        return AnnifConfigCFG(filename)
