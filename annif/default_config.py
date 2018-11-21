#!/usr/bin/env python3

"""
A configuration module, where "Config" is a default configuration and the other
classes are different configuration profiles overriding default settings.
"""

import os


class Config(object):
    DEBUG = False
    TESTING = False
    PROJECTS_FILE = os.environ.get('ANNIF_PROJECTS', default='projects.cfg')
    DATADIR = os.environ.get('ANNIF_DATADIR', default='data')
    INITIALIZE_PROJECTS = False


class ProductionConfig(Config):
    INITIALIZE_PROJECTS = True


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    PROJECTS_FILE = 'tests/projects.cfg'
    DATADIR = 'tests/data'


class TestingInitializeConfig(TestingConfig):
    INITIALIZE_PROJECTS = True


class TestingNoProjectsConfig(TestingConfig):
    PROJECTS_FILE = 'tests/notfound.cfg'
