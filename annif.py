#!/usr/bin/env python3

import click
import connexion
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient, CatClient

es = Elasticsearch()
index = IndicesClient(es)
CAT = CatClient(es)

# 'annif' here is the Connexion application that has a normal Flask app as a
# property (annif.app)

annif = connexion.App(__name__, specification_dir='swagger/')

annif.app.config.from_object('config.Config')

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


def format_index_name(project_id):
    """
    Return an index name formatted like annif-project.
    """
    return "{0}-{1}".format(annif.app.config['INDEX_NAME'], project_id)


def list_orphan_indices():
    """
    Returns a list containing names of orphaned indices.
    """
    indices = [x.split()[2] for x in CAT.indices().split('\n') if len(x) > 0]
    return [x for x in indices if x.startswith(annif.app.config['INDEX_NAME'])]


def init():
    """
    Generate the Elasticsearch repository for projects.

    Usage: annif init
    """
    if index.exists(annif.app.config['INDEX_NAME']):
        index.delete(annif.app.config['INDEX_NAME'])

    # When the repository is initialized, check also if any orphaned indices
    # (= indices starting with INDEX_NAME) are found and remove them.

    for i in list_orphan_indices():
        index.delete(i)

    es.indices.create(index=annif.app.config['INDEX_NAME'],
                      body=projectIndexConf)
    return 'Initialized project index \'{0}\'.'.format(
            annif.app.config['INDEX_NAME'])


def list_projects():
    """
    List available projects.

    Usage: annif list-projects

    REST equivalent: GET /projects/
    """

    doc = {'size': 1000, 'query': {'match_all': {}}}

    return [x['_source'] for x in es.search(
        index=annif.app.config['INDEX_NAME'],
        doc_type='project',
        body=doc)['hits']['hits']]


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


def show_project(project_id):
    """
    Show project information.

    Usage: annif show-project <project_id>

    REST equivalent:

    GET /projects/<project_id>
    """
    result = es.search(index=annif.app.config['INDEX_NAME'],
                       doc_type='project',
                       body={'query': {'match': {'name': project_id}}})

    if result['hits']['hits']:
        return format_result(result)
    else:
        return "No projects found with id \'{0}\'.".format(project_id)


def add_to_master_index(body):
    """
    Takes a dict containing project information and adds it as a record
    in the 'master index'.

    """
    return es.create(index=annif.app.config['INDEX_NAME'],
                     doc_type='project',
                     id=body['name'],
                     body=body)


def create_project(project_id, language, analyzer):
    """
    Create a new project.

    Usage: annif create-project <project_id> --language <lang> --analyzer
    <analyzer>

    REST equivalent:

    PUT /projects/<project_id>
    """

    proj_indexname = format_index_name(project_id)
    body = {'name': project_id, 'language': language, 'analyzer': analyzer}

    if not all(body.values()):
        return 'Usage: annif create-project <project_id> --language <lang> ' \
               '--analyzer <analyzer>'

    elif index.exists(proj_indexname):
        return '\'{0}\' already exists.'.format(proj_indexname)
    else:
        # Create an index for the project
        index.create(index=proj_indexname)

        # Add the details of the new project to the 'master' index
        add_to_master_index(body)
        return 'Successfully created project \'{0}\'.'.format(project_id)


def drop_project(project_id):
    """
    Delete a project.
    USAGE: annif drop-project <project_id>

    REST equivalent:

    DELETE /projects/<project_id>
    """
    # Delete the index from the 'master' index
    result = es.delete(index=annif.app.config['INDEX_NAME'],
                       doc_type='project', id=project_id)

    print(result)

    # Then delete the project index
    return index.delete(index=format_index_name(project_id))


def list_subjects(project_id):
    """
    Show all subjects for a project.

    USAGE: annif list-subjects <project_id>

    REST equivalent:

    GET /projects/<project_id>/subjects
    """
    pass


def show_subject(project_id, subject_id):
    """
    Show information about a subject.

    USAGE: annif show-subject <project_id> <subject_id>

    REST equivalent:

    GET /projects/<project_id>/subjects/<subject_id>
    """
    pass


def create_subject(project_id, subject_id):
    """
    Create a new subject, or update an existing one.

    annif create-subject <project_id> <subject_id> <subject.txt

    REST equivalent:

    PUT /projects/<project_id>/subjects/<subject_id>
    """
    pass


def load(project_id, directory, clear):
    """
    Load all subjects from a directory.

    USAGE: annif load <project_id> <directory> [--clear=CLEAR]
    """
    pass


def drop_subject(project_id, subject_id):
    """
    Delete a subject.

    USAGE: annif drop-subject <project_id> <subject_id>

    REST equivalent:

    DELETE /projects/<project_id>/subjects/<subject_id>

    """
    pass


def analyze(project_id, maxhits, threshold):
    """"
    Delete a subject.

    USAGE: annif drop-subject <project_id> <subject_id>

    REST equivalent:

    DELETE /projects/<project_id>/subjects/<subject_id>

    """
    pass


##############################################################################
# COMMAND-LINE INTERFACE
# Here are the definitions for command-line (Click) commands for invoking
# the above functions and printing the results to console.
##############################################################################

@annif.app.cli.command('init')
def run_init():
    print(init())


@annif.app.cli.command('list-projects')
def run_list_projects():
    template = "{0: <15}{1: <15}{2: <15}\n"

    formatted = template.format("Project ID", "Language", "Analyzer")
    formatted += str("-" * len(formatted) + "\n")

    for proj in list_projects():
        formatted += template.format(proj['name'], proj['language'],
                                     proj['analyzer'])

    print(formatted)


@annif.app.cli.command('create-project')
@click.argument('project_id')
@click.option('--language')
@click.option('--analyzer')
def run_create_project(project_id, language, analyzer):
    print(create_project(project_id, language, analyzer))


@annif.app.cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    print(show_project(project_id))


@annif.app.cli.command('drop-project')
@click.argument('project_id')
def run_drop_project(project_id):
    print(drop_project(project_id))


@annif.app.cli.command('load')
@click.argument('project_id')
@click.argument('directory')
@click.option('--clear', default=False)
def run_load(project_id, directory, clear):
    print("TODO")


@annif.app.cli.command('list-subjects')
@click.argument('project_id')
def run_list_subjects():
    print("TODO")


@annif.app.cli.command('show-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_show_subject(project_id, subject_id):
    print("TODO")


@annif.app.cli.command('create-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_create_subject(project_id, subject_id):
    print("TODO")


@annif.app.cli.command('drop-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_drop_subject(project_id, subject_id):
    print("TODO")


@annif.app.cli.command('analyze')
@click.option('--maxhits', default=20)
@click.option('--threshold', default=0.9)  # TODO: Check this.
def run_analyze(project_id, maxhits, threshold):
    print(analyze(project_id, maxhits, threshold))

##############################################################################


annif.add_api('annif.yaml')

application = annif.app

if __name__ == "__main__":
    annif.run(port=8080)
