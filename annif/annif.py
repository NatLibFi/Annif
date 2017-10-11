#!/usr/bin/env python3

import click
from flask import Flask
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient, CatClient

es = Elasticsearch()
index = IndicesClient(es)
CAT = CatClient(es)

annif = Flask(__name__)

annif.config.from_object('annif.config.Config')


projectIndexConf = {
        'mappings': {
            'project': {
                'properties': {
                    'name': {
                        'type': 'text'
                        },
                    'language': {
                        'type': 'text'
                        },
                    'analyzer': {
                        'type': 'text'
                        }
                    }
                }
            }
        }


def format_index_name(projectid):
    """
    Return an index name formatted like annif-project.
    """
    return "{0}-{1}".format(annif.config['INDEX_NAME'], projectid)


def list_orphan_indices():
    """
    Returns a list containing names of orphaned indices.
    """
    indices = [x.split()[2] for x in CAT.indices().split('\n') if len(x) > 0]
    return [x for x in indices if x.startswith(annif.config['INDEX_NAME'])]


@annif.cli.command('init')
def init():
    """
    Generate the Elasticsearch repository for projects.

    Usage: annif init
    """
    if index.exists(annif.config['INDEX_NAME']):
        index.delete(annif.config['INDEX_NAME'])

    # When the repository is initialized, check also if any orphaned indices
    # (= indices starting with INDEX_NAME) are found and remove them.

    for i in list_orphan_indices():
        index.delete(i)

    print('Initialized project index \'{0}\'.'.format(
        annif.config['INDEX_NAME']))
    return es.indices.create(index=annif.config['INDEX_NAME'],
                             body=projectIndexConf)


@annif.cli.command('list-projects')
def list_projects():
    """
    List available projects.

    Usage: annif list-projects

    REST equivalent: GET /projects/
    """

    doc = {'size': 1000, 'query': {'match_all': {}}}

    template = "{0: <15}{1: <15}{2: <15}\n"

    formatted = template.format("Project ID", "Language", "Analyzer")
    formatted += str("-" * len(formatted) + "\n")

    projects = [x['_source'] for x in es.search(
        index=annif.config['INDEX_NAME'],
        doc_type='project',
        body=doc)['hits']['hits']]

    for proj in projects:
        formatted += template.format(proj['name'], proj['language'],
                                     proj['analyzer'])

    print(formatted)


def format_result(result):
    """
    A helper function for show_project. Takes a dict containing project
    information as returned by ES client and returns a human-readable
    string representation formatted as follows:

    Project ID:    testproj
    Language:      fi
    Analyzer       finglish

    """
    template = "{0:<15}{1}\n"
    content = result['hits']['hits'][0]
    formatted = template.format('Project ID:', content['_source']['name'])
    formatted += template.format('Language:', content['_source']['language'])
    formatted += template.format('Analyzer', content['_source']['analyzer'])
    return formatted


@annif.cli.command('show-project')
@click.argument('projectid')
def show_project(projectid):
    """
    Show project information.

    Usage: annif show-project <projectId>

    REST equivalent:

    GET /projects/<projectId>
    """
    result = es.search(index=annif.config['INDEX_NAME'],
                       doc_type='project',
                       body={'query': {'match': {'name': projectid}}})

    if result['hits']['hits']:
        print(format_result(result))
    else:
        print("No projects found with id \'{0}\'.".format(projectid))


@annif.cli.command('create-project')
@click.argument('projectid')
@click.option('--language')
@click.option('--analyzer')
def create_project(projectid, language, analyzer):
    """
    Create a new project.

    Usage: annif create-project <projectId> --language <lang> --analyzer
    <analyzer>

    REST equivalent:

    PUT /projects/<projectId>
    """

    if not projectid or not language or not analyzer:
        print('Usage: annif create-project <projectId> --language <lang> '
              '--analyzer <analyzer>')
    elif index.exists(proj_indexname):
        print('Index \'{0}\' already exists.'
                .format(format_index_name(projectid)))
    else:
        # Create an index for the project
        index.create(index=format_index_name(projectid))

        # Add the details of the new project to the 'master' index
        es.create(index=annif.config['INDEX_NAME'],
                  doc_type='project', id=projectid,
                  body={'name': projectid, 'language': language,
                        'analyzer': analyzer})
        print('Successfully created project \'{0}\'.'.format(projectid))


@annif.cli.command('drop-project')
@click.argument('projectid')
def drop_project(projectid):
    """
    Delete a project.
    USAGE: annif drop-project <projectid>

    REST equivalent:

    DELETE /projects/<projectid>
    """
    # Delete the index from the 'master' index
    result = es.delete(index=annif.config['INDEX_NAME'],
                       doc_type='project', id=projectid)

    print(result)

    # Then delete the project index
    result = index.delete(index=format_index_name(projectid))
    print(result)


@annif.cli.command('list-subjects')
@click.argument('projectid')
def list_subjects(projectid):
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
def show_subject(projectid, subjectid):
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
def create_subject(projectid, subjectid):
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
def drop_subject(projectid, subjectid):
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
