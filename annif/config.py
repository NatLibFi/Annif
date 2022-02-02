"""Configuration file handling"""


import configparser
import tomli
import annif
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
                raise ConfigurationException(err)

    @property
    def project_ids(self):
        return self._config.keys()

    def __getitem__(self, key):
        return self._config[key]
