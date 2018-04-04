"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import logging
import sys
import click
import click_log
from flask.cli import FlaskGroup
import annif
import annif.corpus
import annif.eval
import annif.project
from annif.hit import HitFilter
from annif import logger

click_log.basic_config(logger)

cli = FlaskGroup(create_app=annif.create_app)


def get_project(project_id):
    """
    Helper function to get a project by ID and bail out if it doesn't exist"""
    try:
        return annif.project.get_project(project_id)
    except ValueError:
        click.echo(
            "No projects found with id \'{0}\'.".format(project_id),
            err=True)
        sys.exit(1)


def parse_backend_params(backend_param):
    """Parse a list of backend parameters given with the --backend-param
    option into a nested dict structure"""
    backend_params = collections.defaultdict(dict)
    for beparam in backend_param:
        backend, param = beparam.split('.', 1)
        key, val = param.split('=', 1)
        backend_params[backend][key] = val
    return backend_params


@cli.command('list-projects')
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


@cli.command('show-project')
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


@cli.command('load')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory')
def run_load(project_id, directory):
    proj = get_project(project_id)
    subjects = annif.corpus.SubjectDirectory(directory)
    proj.load_subjects(subjects)


@cli.command('list-subjects')
@click.argument('project_id')
def run_list_subjects():
    click.echo("TODO")


@cli.command('show-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_show_subject(project_id, subject_id):
    click.echo("TODO")


@cli.command('create-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_create_subject(project_id, subject_id):
    click.echo("TODO")


@cli.command('drop-subject')
@click.argument('project_id')
@click.argument('subject_id')
def run_drop_subject(project_id, subject_id):
    click.echo("TODO")


@cli.command('analyze')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_analyze(project_id, limit, threshold, backend_param):
    """"
    Analyze a document.

    USAGE: annif analyze <project_id> [--limit=N] [--threshold=N] <document.txt
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit, threshold)
    hits = hit_filter(project.analyze(text, backend_params))
    for hit in hits:
        click.echo("{}\t<{}>\t{}".format(hit.score, hit.uri, hit.label))


@cli.command('eval')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('subject_file')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_eval(project_id, subject_file, limit, threshold, backend_param):
    """"
    Evaluate the analysis result for a document against a gold standard
    given in a subject file.

    USAGE: annif eval <project_id> <subject_file> [--limit=N]
           [--threshold=N] <document.txt
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit=limit, threshold=threshold)
    hits = hit_filter(project.analyze(text, backend_params))
    with open(subject_file) as subjfile:
        gold_subjects = annif.corpus.SubjectSet(subjfile.read())

    template = "{0:<20}\t{1}"
    for metric, result, merge_function in annif.eval.evaluate_hits(
            hits, gold_subjects):
        click.echo(template.format(metric + ":", result))


@cli.command('evaldir')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory')
@click.option('--limit', default=10)
@click.option('--threshold', default=0.0)
@click.option('--backend-param', '-b', multiple=True)
def run_evaldir(project_id, directory, limit, threshold, backend_param):
    """"
    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files.

    USAGE: annif evaldir <project_id> <directory> [--limit=N]
           [--threshold=N]
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    hit_filter = HitFilter(limit=limit, threshold=threshold)
    eval_batch = annif.eval.EvaluationBatch()
    for docfilename, subjectfilename in annif.corpus.DocumentDirectory(
            directory, require_subjects=True):
        with open(docfilename) as docfile:
            text = docfile.read()
        hits = hit_filter(project.analyze(text, backend_params))
        with open(subjectfilename) as subjfile:
            gold_subjects = annif.corpus.SubjectSet(subjfile.read())
        eval_batch.evaluate(hits, gold_subjects)

    template = "{0:<20}\t{1}"
    for metric, score in eval_batch.results().items():
        click.echo(template.format(metric + ":", score))


@cli.command('optimize')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory')
@click.option('--backend-param', '-b', multiple=True)
def run_optimize(project_id, directory, backend_param):
    """"
    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files. Test different limit/threshold
    values and report the precision, recall and F-measure of each combination
    of settings.

    USAGE: annif optimize <project_id> <directory>
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    filter_batches = collections.OrderedDict()
    for limit in range(1, 16):
        for threshold in [i * 0.05 for i in range(20)]:
            hit_filter = HitFilter(limit, threshold)
            batch = annif.eval.EvaluationBatch()
            filter_batches[(limit, threshold)] = (hit_filter, batch)

    for docfilename, subjectfilename in annif.corpus.DocumentDirectory(
            directory, require_subjects=True):
        with open(docfilename) as docfile:
            text = docfile.read()
        hits = project.analyze(text, backend_params)
        with open(subjectfilename) as subjfile:
            gold_subjects = annif.corpus.SubjectSet(subjfile.read())
        for hit_filter, batch in filter_batches.values():
            batch.evaluate(hit_filter(hits), gold_subjects)

    click.echo("\t".join(('Limit', 'Thresh.', 'Prec.', 'Rec.', 'F-meas.')))

    best_scores = collections.defaultdict(float)
    best_params = {}

    template = "{:d}\t{:.02f}\t{:.04f}\t{:.04f}\t{:.04f}"
    for params, filter_batch in filter_batches.items():
        results = filter_batch[1].results()
        for metric, score in results.items():
            if score > best_scores[metric]:
                best_scores[metric] = score
                best_params[metric] = params
        click.echo(
            template.format(
                params[0],
                params[1],
                results['Precision'],
                results['Recall'],
                results['F-measure']))

    click.echo()
    template2 = "Best {}:\t{:.04f}\tLimit: {:d}\tThreshold: {:.02f}"
    for metric in ('Precision', 'Recall', 'F-measure', 'NDCG@5', 'NDCG@10'):
        click.echo(
            template2.format(
                metric,
                best_scores[metric],
                best_params[metric][0],
                best_params[metric][1]))


if __name__ == '__main__':
    cli()
