"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import sys
import click
import annif
import annif.operations


@annif.cxapp.app.cli.command('list-projects')
def run_list_projects():
    template = "{0: <15}{1: <15}{2: <15}\n"

    formatted = template.format("Project ID", "Language", "Analyzer")
    formatted += str("-" * len(formatted) + "\n")

    for proj in annif.operations.list_projects():
        formatted += template.format(proj.project_id, proj.language,
                                     proj.analyzer)

    print(formatted)


@annif.cxapp.app.cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    """
    Takes a dict containing project information as returned by ES client
    and returns a human-readable string representation formatted as follows:

    Project ID:    testproj
    Language:      fi
    Analyzer       finglish

    """

    proj = annif.operations.show_project(project_id)
    if not isinstance(proj, str):
        formatted = ""
        template = "{0:<15}{1}\n"

        formatted = template.format('Project ID:', proj.project_id)
        formatted += template.format('Language:', proj.language)
        formatted += template.format('Analyzer', proj.analyzer)
        print(formatted)
    else:
        print(proj)


@annif.cxapp.app.cli.command('load')
@click.argument('project_id')
@click.argument('directory')
@click.option('--clear', default=False)
def run_load(project_id, directory, clear):
    print("TODO")


@annif.cxapp.app.cli.command('list-subjects')
@click.argument('project_id')
def run_list_subjects():
    print("TODO")


@annif.cxapp.app.cli.command('show-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_show_subject(project_id, subject_id):
    print("TODO")


@annif.cxapp.app.cli.command('create-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_create_subject(project_id, subject_id):
    print("TODO")


@annif.cxapp.app.cli.command('drop-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_drop_subject(project_id, subject_id):
    print("TODO")


@annif.cxapp.app.cli.command('analyze')
@click.argument('project_id')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_analyze(project_id, limit, threshold):
    """"
    Analyze a document.

    USAGE: annif analyze <project_id> [--limit=N] [--threshold=N] <document.txt

    REST equivalent:

    POST /projects/<project_id>/analyze

    """
    text = sys.stdin.read()
    hits = annif.operations.analyze(project_id, text, limit, threshold)

    for hit in hits:
        print("{}\t{}\t{}".format(hit.score, hit.uri, hit.label))
