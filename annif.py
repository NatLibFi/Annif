#!/usr/bin/env python3

import click
import connexion
from flask import Flask

annif = Flask(__name__)

annif.config.from_object('config.DevelopmentConfig')

@annif.cli.command()
def setup():
    return 'Setting up ElasticSearch'

@annif.route('/')
def start():
    return 'Started application'

if __name__ == "__main__":
    annif.run(port=8000)
