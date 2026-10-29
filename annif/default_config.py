#!/usr/bin/env python3

"""
A configuration module, where "Config" is a default configuration and the other
classes are different configuration profiles overriding default settings.
"""

import os


class Config(object):
    DEBUG = False
    TESTING = False
    PROJECTS_CONFIG_PATH = os.environ.get("ANNIF_PROJECTS", default="")
    DATADIR = os.environ.get("ANNIF_DATADIR", default="data")
    INITIALIZE_PROJECTS = False
    MAX_FORM_MEMORY_SIZE = int(
        os.environ.get("ANNIF_MAX_FORM_MEMORY_SIZE", default=20_000_000)
    )
    MAX_CONTENT_LENGTH = int(
        os.environ.get("ANNIF_MAX_CONTENT_LENGTH", default=20_000_000)
    )


class ProductionConfig(Config):
    INITIALIZE_PROJECTS = True


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    PROJECTS_CONFIG_PATH = "tests/projects.cfg"
    DATADIR = "tests/data"
    # keep the form field limit below the body limit so that oversized form
    # fields are rejected by the form data validator instead of the body
    # limit middleware
    MAX_CONTENT_LENGTH = 10_000_000
    MAX_FORM_MEMORY_SIZE = 2_000_000


class TestingInitializeConfig(TestingConfig):
    INITIALIZE_PROJECTS = True


class TestingNoProjectsConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = "tests/notfound.cfg"


class TestingInvalidProjectsConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = "tests/projects_invalid.cfg"


class TestingInvalid2ProjectsConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = "tests/projects_invalid2.cfg"


class TestingTOMLConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = "tests/projects.toml"


class TestingDirectoryConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = "tests/projects.d"
