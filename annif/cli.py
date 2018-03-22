"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import logging
import statistics
import sys
import click
import click_log
import annif
import annif.corpus
import annif.eval
import annif.project
from annif import logger

click_log.basic_config(logger)


def get_project(project_id):
    """Helper function to get a project by ID and bail out if it doesn't exist"""
    try:
        return annif.project.get_project(project_id)
    except ValueError:
        click.echo(
            "No projects found with id \'{0}\'.".format(project_id),
            err=True)
        sys.exit(1)


@annif.cxapp.app.cli.command('list-projects')
def run_list_projects():
    """
    List available projects.

    Usage: annif list-projects
    """

    template = "{0: <15}{1: <15}"

    header = template.format("Project ID", "Language")
    click.echo(header)
    click.echo("-" * len(header))

    for proj in annif.project.get_projects().values():
        click.echo(template.format(proj.project_id, proj.language))


@annif.cxapp.app.cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    """
    Show project information.

    Usage: annif show-project <project_id>

    Outputs a human-readable string representation formatted as follows:

    Project ID:    testproj
    Language:      fi
    """

    proj = get_project(project_id)

    template = "{0:<15}{1}"

    click.echo(template.format('Project ID:', proj.project_id))
    click.echo(template.format('Language:', proj.language))


@annif.cxapp.app.cli.command('load')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory')
def run_load(project_id, directory):
    proj = get_project(project_id)
    subjects = annif.corpus.SubjectDirectory(directory)
    proj.load_subjects(subjects)


@annif.cxapp.app.cli.command('list-subjects')
@click.argument('project_id')
def run_list_subjects():
    click.echo("TODO")


@annif.cxapp.app.cli.command('show-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_show_subject(project_id, subject_id):
    click.echo("TODO")


@annif.cxapp.app.cli.command('create-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_create_subject(project_id, subject_id):
    click.echo("TODO")


@annif.cxapp.app.cli.command('drop-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_drop_subject(project_id, subject_id):
    click.echo("TODO")


@annif.cxapp.app.cli.command('analyze')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_analyze(project_id, limit, threshold):
    """"
    Analyze a document.

    USAGE: annif analyze <project_id> [--limit=N] [--threshold=N] <document.txt
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    hits = project.analyze(text, limit, threshold)
    for hit in hits:
        click.echo("{}\t<{}>\t{}".format(hit.score, hit.uri, hit.label))


@annif.cxapp.app.cli.command('eval')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('subject_file')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_eval(project_id, subject_file, limit, threshold):
    """"
    Evaluate the analysis result for a document against a gold standard
    given in a subject file.

    USAGE: annif eval <project_id> <subject_file> [--limit=N]
           [--threshold=N] <document.txt
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    hits = project.analyze(text, limit, threshold)
    with open(subject_file) as subjfile:
        gold_subjects = annif.corpus.SubjectSet(subjfile.read())

    template = "{0:<10}\t{1}"
    for metric, result in annif.eval.evaluate_hits(hits, gold_subjects):
        click.echo(template.format(metric + ":", result))


@annif.cxapp.app.cli.command('evaldir')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
def run_evaldir(project_id, directory, limit, threshold):
    """"
    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files.

    USAGE: annif evaldir <project_id> <directory> [--limit=N]
           [--threshold=N]
    """
    project = get_project(project_id)

    measures = collections.OrderedDict()
    for docfilename, subjectfilename in annif.corpus.DocumentDirectory(
            directory, require_subjects=True):
        with open(docfilename) as docfile:
            text = docfile.read()
        hits = project.analyze(text, limit, threshold)
        with open(subjectfilename) as subjfile:
            gold_subjects = annif.corpus.SubjectSet(subjfile.read())

        for metric, result in annif.eval.evaluate_hits(hits, gold_subjects):
            measures.setdefault(metric, [])
            measures[metric].append(result)

    template = "{0:<10}\t{1}"
    for metric, results in measures.items():
        click.echo(template.format(metric + ":", statistics.mean(results)))
