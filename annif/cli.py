"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import os.path
import re
import sys
import click
import click_log
from flask.cli import FlaskGroup
import annif
import annif.corpus
import annif.eval
import annif.project
from annif.hit import HitFilter

logger = annif.logger
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


def open_documents(paths):
    """Helper function to open a document corpus from a list of pathnames,
    each of which is either a TSV file or a directory of TXT files. The
    corpus will be returned as an instance of DocumentCorpus."""

    def open_doc_path(path):
        """open a single path and return it as a DocumentCorpus"""
        if os.path.isdir(path):
            return annif.corpus.DocumentDirectory(path, require_subjects=True)
        return annif.corpus.DocumentFile(path)

    if len(paths) > 1:
        corpora = [open_doc_path(path) for path in paths]
        docs = annif.corpus.CombinedCorpus(corpora)
    else:
        docs = open_doc_path(paths[0])
    return docs


def parse_backend_params(backend_param):
    """Parse a list of backend parameters given with the --backend-param
    option into a nested dict structure"""
    backend_params = collections.defaultdict(dict)
    for beparam in backend_param:
        backend, param = beparam.split('.', 1)
        key, val = param.split('=', 1)
        backend_params[backend][key] = val
    return backend_params


def generate_filter_batches(subjects):
    filter_batches = collections.OrderedDict()
    for limit in range(1, 16):
        for threshold in [i * 0.05 for i in range(20)]:
            hit_filter = HitFilter(limit, threshold)
            batch = annif.eval.EvaluationBatch(subjects)
            filter_batches[(limit, threshold)] = (hit_filter, batch)
    return filter_batches


@cli.command('list-projects')
def run_list_projects():
    """
    List available projects.
    """

    template = "{0: <15}{1: <30}{2: <15}"

    header = template.format("Project ID", "Project Name", "Language")
    click.echo(header)
    click.echo("-" * len(header))

    for proj in annif.project.get_projects().values():
        click.echo(template.format(proj.project_id, proj.name, proj.language))


@cli.command('show-project')
@click.argument('project_id')
def run_show_project(project_id):
    """
    Show information about a project.
    """

    proj = get_project(project_id)

    template = "{0:<20}{1}"

    click.echo(template.format('Project ID:', proj.project_id))
    click.echo(template.format('Project Name:', proj.project_id))
    click.echo(template.format('Language:', proj.language))


@cli.command('loadvoc')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('subjectfile', type=click.Path(dir_okay=False))
def run_loadvoc(project_id, subjectfile):
    """
    Load a vocabulary for a project.
    """
    proj = get_project(project_id)
    if annif.corpus.SubjectFileSKOS.is_rdf_file(subjectfile):
        # SKOS/RDF file supported by rdflib
        subjects = annif.corpus.SubjectFileSKOS(subjectfile, proj.language)
    else:
        # probably a TSV file
        subjects = annif.corpus.SubjectFileTSV(subjectfile)
    proj.vocab.load_vocabulary(subjects)


@cli.command('train')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('paths', type=click.Path(), nargs=-1)
def run_train(project_id, paths):
    """
    Train a project on a collection of documents.
    """
    proj = get_project(project_id)
    documents = open_documents(paths)
    proj.load_documents(documents)


@cli.command('analyze')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.option('--limit', default=10, help='Maximum number of subjects')
@click.option('--threshold', default=0.0, help='Minimum score threshold')
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
def run_analyze(project_id, limit, threshold, backend_param):
    """
    Analyze a single document from standard input.
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit, threshold)
    hits = hit_filter(project.analyze(text, backend_params))
    for hit in hits:
        click.echo("<{}>\t{}\t{}".format(hit.uri, hit.label, hit.score))


@cli.command('analyzedir')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('directory', type=click.Path(file_okay=False))
@click.option(
    '--suffix',
    default='.annif',
    help='File name suffix for result files')
@click.option('--force/--no-force', default=False,
              help='Force overwriting of existing result files')
@click.option('--limit', default=10, help='Maximum number of subjects')
@click.option('--threshold', default=0.0, help='Minimum score threshold')
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
def run_analyzedir(project_id, directory, suffix, force,
                   limit, threshold, backend_param):
    """
    Analyze a directory with documents. Write the results in TSV files
    with the given suffix.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)
    hit_filter = HitFilter(limit, threshold)

    for docfilename, dummy_subjectfn in annif.corpus.DocumentDirectory(
            directory, require_subjects=False):
        with open(docfilename) as docfile:
            text = docfile.read()
        subjectfilename = re.sub(r'\.txt$', suffix, docfilename)
        if os.path.exists(subjectfilename) and not force:
            click.echo(
                "Not overwriting {} (use --force to override)".format(
                    subjectfilename))
            continue
        with open(subjectfilename, 'w') as subjfile:
            results = project.analyze(text, backend_params)
            for hit in hit_filter(results):
                line = "<{}>\t{}\t{}".format(hit.uri, hit.label, hit.score)
                click.echo(line, file=subjfile)


@cli.command('eval')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('paths', type=click.Path(), nargs=-1)
@click.option('--limit', default=10, help='Maximum number of subjects')
@click.option('--threshold', default=0.0, help='Minimum score threshold')
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
def run_eval(project_id, paths, limit, threshold, backend_param):
    """
    Analyze documents and evaluate the result.

    Compare the results of automated indexing against a gold standard. The
    path may be either a TSV file with short documents or a directory with
    documents in separate files.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    hit_filter = HitFilter(limit=limit, threshold=threshold)
    eval_batch = annif.eval.EvaluationBatch(project.subjects)

    docs = open_documents(paths)
    for doc in docs.documents:
        results = project.analyze(doc.text, backend_params)
        hits = hit_filter(results)
        eval_batch.evaluate(hits,
                            annif.corpus.SubjectSet((doc.uris, doc.labels)))

    template = "{0:<20}\t{1}"
    for metric, score in eval_batch.results().items():
        click.echo(template.format(metric + ":", score))


@cli.command('optimize')
@click_log.simple_verbosity_option(logger)
@click.argument('project_id')
@click.argument('paths', type=click.Path(), nargs=-1)
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
def run_optimize(project_id, paths, backend_param):
    """
    Analyze documents, testing multiple limits and thresholds.

    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files. Test different limit/threshold
    values and report the precision, recall and F-measure of each combination
    of settings.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    filter_batches = generate_filter_batches(project.subjects)

    ndocs = 0
    docs = open_documents(paths)
    for doc in docs.documents:
        hits = project.analyze(doc.text, backend_params)
        gold_subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
        for hit_filter, batch in filter_batches.values():
            batch.evaluate(hit_filter(hits), gold_subjects)
        ndocs += 1

    click.echo("\t".join(('Limit', 'Thresh.', 'Prec.', 'Rec.', 'F1')))

    best_scores = collections.defaultdict(float)
    best_params = {}

    template = "{:d}\t{:.02f}\t{:.04f}\t{:.04f}\t{:.04f}"
    for params, filter_batch in filter_batches.items():
        results = filter_batch[1].results()
        for metric, score in results.items():
            if score >= best_scores[metric]:
                best_scores[metric] = score
                best_params[metric] = params
        click.echo(
            template.format(
                params[0],
                params[1],
                results['Precision (doc avg)'],
                results['Recall (doc avg)'],
                results['F1 score (doc avg)']))

    click.echo()
    template2 = "Best {:>19}: {:.04f}\tLimit: {:d}\tThreshold: {:.02f}"
    for metric in ('Precision (doc avg)',
                   'Recall (doc avg)',
                   'F1 score (doc avg)',
                   'NDCG@5',
                   'NDCG@10'):
        click.echo(
            template2.format(
                metric,
                best_scores[metric],
                best_params[metric][0],
                best_params[metric][1]))
    click.echo("Documents evaluated:\t{}".format(ndocs))


if __name__ == '__main__':
    cli()
