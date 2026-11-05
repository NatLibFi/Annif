"""Configuration file handling"""

from __future__ import annotations

import configparser
import copy
import os.path
import tomllib
from glob import glob

import annif
import annif.util
from annif.exception import ConfigurationException
from annif.util import interpolate_env_vars

logger = annif.logger


class AnnifConfigCFG:
    """Class for reading configuration in CFG/INI format"""

    def __init__(self, filename: str = None, projstr: str = None) -> None:
        self._config = configparser.ConfigParser()
        self._config.optionxform = annif.util.identity
        self._raw_config = configparser.ConfigParser()
        if filename is not None:
            logger.debug(f"Reading configuration file {filename} in CFG format")
            self._read_config(self._config.read, filename)
        elif projstr is not None:
            logger.debug("Reading configuration from a string in CFG format")
            self._read_config(self._config.read_string, projstr)

    def _read_config(self, read_method, source):
        encoding = "utf-8-sig"
        try:
            read_method(source, encoding)
        except (
            configparser.DuplicateOptionError,
            configparser.DuplicateSectionError,
        ) as err:
            raise ConfigurationException(err.message)
        # Keep a copy of the raw configuration, without the environment
        # variable placeholders expanded, for use when the configuration
        # is exported (e.g. to a Hugging Face Hub repository), so that
        # secrets are not included in the exported configuration.
        self._raw_config = copy.deepcopy(self._config)
        self._interpolate_env_vars()

    def _interpolate_env_vars(self) -> None:
        """Replace ${VAR} placeholders in all option values with the value
        of the corresponding environment variable."""
        for section in self._config.sections():
            for option, value in list(self._config[section].items()):
                resolved = interpolate_env_vars(value, section, option)
                # Escape percent signs so that ConfigParser's interpolation
                # does not misinterpret them when the value is read back
                self._config[section][option] = resolved.replace("%", "%%")
        # The values in the DEFAULT section may contain ConfigParser
        # interpolation templates (e.g. '%(language)s'), which are only
        # valid in the context of a specific section. Therefore, the
        # environment placeholders are replaced in the raw values, and no
        # escaping is done, in order to preserve the templates.
        for option, value in self._config.defaults().items():
            self._config[configparser.DEFAULTSECT][option] = interpolate_env_vars(
                value, configparser.DEFAULTSECT, option, escape_percent=True
            )

    @property
    def project_ids(self) -> list[str]:
        return self._config.sections()

    def __getitem__(self, key: str) -> configparser.SectionProxy:
        return self._config[key]

    def get_raw_config(self, key: str) -> dict[str, str]:
        """Return the raw configuration of the given project, without
        environment variable placeholders expanded, as a dict of raw
        option values."""
        return {
            option: self._raw_config[key].get(option, raw=True)
            for option in self._raw_config[key]
        }


class AnnifConfigTOML:
    """Class for reading configuration in TOML format"""

    def __init__(self, filename: str) -> None:
        with open(filename, "rb") as projf:
            try:
                logger.debug(f"Reading configuration file {filename} in TOML format")
                self._config = tomllib.load(projf)
            except tomllib.TOMLDecodeError as err:
                raise ConfigurationException(
                    f"Parsing TOML file '{filename}' failed: {err}"
                )
        # Keep a copy of the raw configuration, without the environment
        # variable placeholders expanded, for use when the configuration
        # is exported (e.g. to a Hugging Face Hub repository), so that
        # secrets are not included in the exported configuration.
        self._raw_config = copy.deepcopy(self._config)
        self._interpolate_env_vars()

    def _interpolate_env_vars(self) -> None:
        """Replace ${VAR} placeholders in all option values with the value
        of the corresponding environment variable."""
        for project_id, options in self._config.items():
            for option, value in options.items():
                if isinstance(value, str):
                    options[option] = interpolate_env_vars(value, project_id, option)

    @property
    def project_ids(self):
        return self._config.keys()

    def __getitem__(self, key: str) -> dict[str, str]:
        return self._config[key]

    def get_raw_config(self, key: str) -> dict[str, str]:
        """Return the raw configuration of the given project, without
        environment variable placeholders expanded, as a dict of raw
        option values."""
        return dict(self._raw_config[key])


class AnnifConfigDirectory:
    """Class for reading configuration from directory"""

    def __init__(self, directory: str) -> None:
        files = glob(os.path.join(directory, "*.cfg"))
        files.extend(glob(os.path.join(directory, "*.toml")))
        logger.debug(f"Reading configuration files in directory {directory}")

        self._config = dict()
        self._source_configs = dict()
        for file in sorted(files):
            source_config = parse_config(file)
            for proj_id in source_config.project_ids:
                self._check_duplicate_project_ids(proj_id, file)
                self._config[proj_id] = source_config[proj_id]
                self._source_configs[proj_id] = source_config

    def _check_duplicate_project_ids(self, proj_id: str, file: str) -> None:
        if proj_id in self._config:
            # Error message resembles configparser's DuplicateSection message
            raise ConfigurationException(
                f'While reading from "{file}": project ID "{proj_id}" already '
                "exists in another configuration file in the directory."
            )

    @property
    def project_ids(self):
        return self._config.keys()

    def __getitem__(self, key: str) -> dict[str, str] | configparser.SectionProxy:
        return self._config[key]

    def get_raw_config(self, key: str) -> dict[str, str]:
        """Return the raw configuration of the given project, without
        environment variable placeholders expanded, as a dict of raw
        option values."""
        return self._source_configs[key].get_raw_config(key)


def check_config(projects_config_path: str) -> str | None:
    if os.path.exists(projects_config_path):
        return projects_config_path
    else:
        logger.warning(
            "Project configuration file or directory "
            + f'"{projects_config_path}" is missing. Please provide one. '
            + "You can set the path to the project configuration "
            + "using the ANNIF_PROJECTS environment "
            + 'variable or the command-line option "--projects".'
        )
        return None


def find_config() -> str | None:
    for path in ("projects.cfg", "projects.toml", "projects.d"):
        if os.path.exists(path):
            return path

    logger.warning(
        "Could not find project configuration "
        + '"projects.cfg", "projects.toml" or "projects.d". '
        + "You can set the path to the project configuration "
        + "using the ANNIF_PROJECTS environment "
        + 'variable or the command-line option "--projects".'
    )
    return None


def parse_config(
    projects_config_path: str,
) -> AnnifConfigDirectory | AnnifConfigCFG | AnnifConfigTOML | None:
    if projects_config_path:
        projects_config_path = check_config(projects_config_path)
    else:
        projects_config_path = find_config()

    if not projects_config_path:  # not found
        return None

    if os.path.isdir(projects_config_path):
        return AnnifConfigDirectory(projects_config_path)
    elif projects_config_path.endswith(".toml"):  # TOML format
        return AnnifConfigTOML(projects_config_path)
    else:  # classic CFG/INI style format
        return AnnifConfigCFG(projects_config_path)
