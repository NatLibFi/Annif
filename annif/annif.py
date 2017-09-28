#!/usr/bin/env python3

import click
import connexion
from flask import Flask
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient

es = Elasticsearch()
index = IndicesClient(es)

annif = Flask(__name__)

# annif.config.from_object('config.DevelopmentConfig')


def createProjectType(projectId, language, analyzer):
    return {
            'projectId': projectId,
            'language': language,
            'analyzer': analyzer
            }


INDEX_NAME = 'annif'


@annif.cli.command('init')
def init():
    """
    Generate the Elasticsearch repository for projects.
    Usage: annif init
    """
    if index.exists(INDEX_NAME):
        index.delete(INDEX_NAME)
    response = es.indices.create(index=INDEX_NAME, ignore=400)
    print(response)


@annif.cli.command('list-projects')
def listprojects():
    """
    List available projects.

    Usage: annif list-projects

    REST equivalent: GET /projects/
    """
    # projs = es.search(index=INDEX_NAME)['hits']
    pass


@annif.cli.command('show-project')
@click.argument('projectid')
def showProject(projectid):
    """
    Show project information.

    Usage: annif show-project <projectId>

    REST equivalent:

    GET /projects/<projectId>
    """
    pass


@annif.cli.command('create-project')
@click.argument('projectid')
@click.option('--language')
@click.option('--analyzer')
def createProject(projectid, language, analyzer):
    """
    Create a new project.

    Usage: annif create-project <projectId> --language <lang> --analyzer
    <analyzer>

    REST equivalent:

    PUT /projects/<projectId>
    """
    pass


@annif.cli.command('drop-project')
@click.argument('projectid')
def dropProject(projectid):
    """
    Delete a project.
    USAGE: annif drop-project <projectid>

    REST equivalent:

    DELETE /projects/<projectid>
    """
    pass


@annif.cli.command('list-subjects')
@click.argument('projectid')
def listSubjects(projectid):
    """
    Show all subjects for a project.

    USAGE: annif list-subjects <projectid>

    REST equivalent:

    GET /projects/<projectid>/subjects
    """
    pass


@annif.cli.command('show-subject')
@click.argument('projectid')
@click.argument('subjectid')
def showSubject(projectid, subjectid):
    """
    Show information about a subject.

    USAGE: annif show-subject <projectid> <subjectid>

    REST equivalent:

    GET /projects/<projectid>/subjects/<subjectid>
    """
    pass


@annif.cli.command('create-subject')
@click.argument('projectid')
@click.argument('subjectid')
def createSubject(projectid, subjectid):
    """
    Create a new subject, or update an existing one.

    annif create-subject <projectid> <subjectid> <subject.txt

    REST equivalent:

    PUT /projects/<projectid>/subjects/<subjectid>
    """
    pass


@annif.cli.command('load')
@click.argument('projectid')
@click.argument('directory')
@click.option('--clear', default=False)
def load(projectid, directory, clear):
    """
    Load all subjects from a directory.

    USAGE: annif load <projectid> <directory> [--clear=CLEAR]
    """
    pass


@annif.cli.command('drop-subject')
@click.argument('projectid')
@click.argument('subjectid')
def dropSubject(projectid, subjectid):
    """
    Delete a subject.

    USAGE: annif drop-subject <projectid> <subjectid>

    REST equivalent:

    DELETE /projects/<projectid>/subjects/<subjectid>

    """
    pass


@annif.cli.command('analyze')
@click.option('--maxhits', default=20)
@click.option('--threshold', default=0.9)  # TODO: Check this.
def analyze(projectid, maxhits, threshold):
    """"
    Delete a subject.

    USAGE: annif drop-subject <projectid> <subjectid>

    REST equivalent:

    DELETE /projects/<projectid>/subjects/<subjectid>

    """
    pass


@annif.route('/')
def start():
    return 'Started application'


if __name__ == "__main__":
    annif.run(port=8000)
