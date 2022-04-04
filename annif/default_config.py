#!/usr/bin/env python3

"""
A configuration module, where "Config" is a default configuration and the other
classes are different configuration profiles overriding default settings.
"""

import os


class Config(object):
    DEBUG = False
    TESTING = False
    PROJECTS_CONFIG_PATH = os.environ.get('ANNIF_PROJECTS', default='')
    DATADIR = os.environ.get('ANNIF_DATADIR', default='data')
    INITIALIZE_PROJECTS = False


class ProductionConfig(Config):
    INITIALIZE_PROJECTS = True


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    PROJECTS_CONFIG_PATH = 'tests/projects.cfg'
    DATADIR = 'tests/data'


class TestingInitializeConfig(TestingConfig):
    INITIALIZE_PROJECTS = True


class TestingNoProjectsConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = 'tests/notfound.cfg'


class TestingInvalidProjectsConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = 'tests/projects_invalid.cfg'


class TestingTOMLConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = 'tests/projects.toml'


class TestingDirectoryConfig(TestingConfig):
    PROJECTS_CONFIG_PATH = 'tests/projects.d'
