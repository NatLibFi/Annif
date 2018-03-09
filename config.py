#!/usr/bin/env python3

"""
A configuration module, where "Config" is a default configuration and the other
classes are different configuration profiles overriding default settings.
"""


class Config(object):
    DEBUG = False
    TESTING = False
    INDEX_NAME = 'annif'
    PROJECTS_FILE = 'projects.cfg'
    BACKENDS_FILE = 'backends.cfg'


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
