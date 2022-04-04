"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import os.path
import re
import sys
import json
import click
import click_log
from flask import current_app
from flask.cli import FlaskGroup, ScriptInfo
import annif
import annif.corpus
import annif.parallel
import annif.project
import annif.registry
from annif.project import Access
from annif.suggestion import SuggestionFilter, ListSuggestionResult
from annif.exception import ConfigurationException, NotSupportedException
from annif.util import metric_code

logger = annif.logger
click_log.basic_config(logger)

cli = FlaskGroup(create_app=annif.create_app, add_version_option=False)
cli = click.version_option(message='%(version)s')(cli)


def get_project(project_id):
    """
    Helper function to get a project by ID and bail out if it doesn't exist"""
    try:
        return annif.registry.get_project(project_id,
                                          min_access=Access.private)
    except ValueError:
        click.echo(
            "No projects found with id \'{0}\'.".format(project_id),
            err=True)
        sys.exit(1)


def open_documents(paths, docs_limit):
    """Helper function to open a document corpus from a list of pathnames,
    each of which is either a TSV file or a directory of TXT files. The
    corpus will be returned as an instance of DocumentCorpus or
    LimitingDocumentCorpus."""

    def open_doc_path(path):
        """open a single path and return it as a DocumentCorpus"""
        if os.path.isdir(path):
            return annif.corpus.DocumentDirectory(path, require_subjects=True)
        return annif.corpus.DocumentFile(path)

    if len(paths) == 0:
        logger.warning('Reading empty file')
        docs = open_doc_path(os.path.devnull)
    elif len(paths) == 1:
        docs = open_doc_path(paths[0])
    else:
        corpora = [open_doc_path(path) for path in paths]
        docs = annif.corpus.CombinedCorpus(corpora)
    if docs_limit is not None:
        docs = annif.corpus.LimitingDocumentCorpus(docs, docs_limit)
    return docs


def parse_backend_params(backend_param, project):
    """Parse a list of backend parameters given with the --backend-param
    option into a nested dict structure"""
    backend_params = collections.defaultdict(dict)
    for beparam in backend_param:
        backend, param = beparam.split('.', 1)
        key, val = param.split('=', 1)
        validate_backend_params(backend, beparam, project)
        backend_params[backend][key] = val
    return backend_params


def validate_backend_params(backend, beparam, project):
    if backend != project.config['backend']:
        raise ConfigurationException(
            'The backend {} in CLI option "-b {}" not matching the project'
            ' backend {}.'
            .format(backend, beparam, project.config['backend']))


BATCH_MAX_LIMIT = 15


def generate_filter_batches(subjects):
    import annif.eval
    filter_batches = collections.OrderedDict()
    for limit in range(1, BATCH_MAX_LIMIT + 1):
        for threshold in [i * 0.05 for i in range(20)]:
            hit_filter = SuggestionFilter(subjects, limit, threshold)
            batch = annif.eval.EvaluationBatch(subjects)
            filter_batches[(limit, threshold)] = (hit_filter, batch)
    return filter_batches


def set_project_config_file_path(ctx, param, value):
    """Override the default path or the path given in env by CLI option"""
    with ctx.ensure_object(ScriptInfo).load_app().app_context():
        if value:
            current_app.config['PROJECTS_CONFIG_PATH'] = value


def common_options(f):
    """Decorator to add common options for all CLI commands"""
    f = click.option(
        '-p', '--projects',
        help='Set path to project configuration file or directory',
        type=click.Path(dir_okay=True, exists=True),
        callback=set_project_config_file_path, expose_value=False,
        is_eager=True)(f)
    return click_log.simple_verbosity_option(logger)(f)


def backend_param_option(f):
    """Decorator to add an option for CLI commands to override BE parameters"""
    return click.option(
        '--backend-param', '-b', multiple=True,
        help='Override backend parameter of the config file. ' +
        'Syntax: "-b <backend>.<parameter>=<value>".')(f)


@cli.command('list-projects')
@common_options
@click_log.simple_verbosity_option(logger, default='ERROR')
def run_list_projects():
    """
    List available projects.
    """

    template = "{0: <25}{1: <45}{2: <10}{3: <7}"
    header = template.format(
        "Project ID", "Project Name", "Language", "Trained")
    click.echo(header)
    click.echo("-" * len(header))
    for proj in annif.registry.get_projects(
            min_access=Access.private).values():
        click.echo(template.format(
            proj.project_id, proj.name, proj.language, str(proj.is_trained)))


@cli.command('show-project')
@click.argument('project_id')
@common_options
def run_show_project(project_id):
    """
    Show information about a project.
    """

    proj = get_project(project_id)
    click.echo(f'Project ID:        {proj.project_id}')
    click.echo(f'Project Name:      {proj.name}')
    click.echo(f'Language:          {proj.language}')
    click.echo(f'Access:            {proj.access.name}')
    click.echo(f'Trained:           {proj.is_trained}')
    click.echo(f'Modification time: {proj.modification_time}')


@cli.command('clear')
@click.argument('project_id')
@common_options
def run_clear_project(project_id):
    """
    Initialize the project to its original, untrained state.
    """
    proj = get_project(project_id)
    proj.remove_model_data()


@cli.command('loadvoc')
@click.argument('project_id')
@click.argument('subjectfile', type=click.Path(exists=True, dir_okay=False))
@click.option('--force', '-f', default=False, is_flag=True,
              help='Replace existing vocabulary completely ' +
                   'instead of updating it')
@common_options
def run_loadvoc(project_id, force, subjectfile):
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
    proj.vocab.load_vocabulary(subjects, proj.language, force=force)


@cli.command('train')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--cached/--no-cached', '-c/-C', default=False,
              help='Reuse preprocessed training data from previous run')
@click.option('--docs-limit', '-d', default=None,
              type=click.IntRange(0, None),
              help='Maximum number of documents to use')
@click.option('--jobs',
              '-j',
              default=0,
              help='Number of parallel jobs (0 means choose automatically)')
@backend_param_option
@common_options
def run_train(project_id, paths, cached, docs_limit, jobs, backend_param):
    """
    Train a project on a collection of documents.
    """
    proj = get_project(project_id)
    backend_params = parse_backend_params(backend_param, proj)
    if cached:
        if len(paths) > 0:
            raise click.UsageError(
                "Corpus paths cannot be given when using --cached option.")
        documents = 'cached'
    else:
        documents = open_documents(paths, docs_limit)
    proj.train(documents, backend_params, jobs)


@cli.command('learn')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--docs-limit', '-d', default=None,
              type=click.IntRange(0, None),
              help='Maximum number of documents to use')
@backend_param_option
@common_options
def run_learn(project_id, paths, docs_limit, backend_param):
    """
    Further train an existing project on a collection of documents.
    """
    proj = get_project(project_id)
    backend_params = parse_backend_params(backend_param, proj)
    documents = open_documents(paths, docs_limit)
    proj.learn(documents, backend_params)


@cli.command('suggest')
@click.argument('project_id')
@click.option('--limit', '-l', default=10, help='Maximum number of subjects')
@click.option('--threshold', '-t', default=0.0, help='Minimum score threshold')
@backend_param_option
@common_options
def run_suggest(project_id, limit, threshold, backend_param):
    """
    Suggest subjects for a single document from standard input.
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param, project)
    hit_filter = SuggestionFilter(project.subjects, limit, threshold)
    hits = hit_filter(project.suggest(text, backend_params))
    for hit in hits.as_list(project.subjects):
        click.echo(
            "<{}>\t{}\t{}".format(
                hit.uri,
                '\t'.join(filter(None, (hit.label, hit.notation))),
                hit.score))


@cli.command('index')
@click.argument('project_id')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option(
    '--suffix',
    '-s',
    default='.annif',
    help='File name suffix for result files')
@click.option('--force/--no-force', '-f/-F', default=False,
              help='Force overwriting of existing result files')
@click.option('--limit', '-l', default=10, help='Maximum number of subjects')
@click.option('--threshold', '-t', default=0.0, help='Minimum score threshold')
@backend_param_option
@common_options
def run_index(project_id, directory, suffix, force,
              limit, threshold, backend_param):
    """
    Index a directory with documents, suggesting subjects for each document.
    Write the results in TSV files with the given suffix.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param, project)
    hit_filter = SuggestionFilter(project.subjects, limit, threshold)

    for docfilename, dummy_subjectfn in annif.corpus.DocumentDirectory(
            directory, require_subjects=False):
        with open(docfilename, encoding='utf-8-sig') as docfile:
            text = docfile.read()
        subjectfilename = re.sub(r'\.txt$', suffix, docfilename)
        if os.path.exists(subjectfilename) and not force:
            click.echo(
                "Not overwriting {} (use --force to override)".format(
                    subjectfilename))
            continue
        with open(subjectfilename, 'w', encoding='utf-8') as subjfile:
            results = project.suggest(text, backend_params)
            for hit in hit_filter(results).as_list(project.subjects):
                line = "<{}>\t{}\t{}".format(
                    hit.uri,
                    '\t'.join(filter(None, (hit.label, hit.notation))),
                    hit.score)
                click.echo(line, file=subjfile)


@cli.command('eval')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--limit', '-l', default=10, help='Maximum number of subjects')
@click.option('--threshold', '-t', default=0.0, help='Minimum score threshold')
@click.option('--docs-limit', '-d', default=None,
              type=click.IntRange(0, None),
              help='Maximum number of documents to use')
@click.option('--metric', '-m', default=[], multiple=True,
              help='Metric to calculate (default: all)')
@click.option(
    '--metrics-file',
    '-M',
    type=click.File(
        'w',
        encoding='utf-8',
        errors='ignore',
        lazy=True),
    help="""Specify file in order to write evaluation metrics in JSON format.
    File directory must exist, existing file will be overwritten.""")
@click.option(
    '--results-file',
    '-r',
    type=click.File(
        'w',
        encoding='utf-8',
        errors='ignore',
        lazy=True),
    help="""Specify file in order to write non-aggregated results per subject.
    File directory must exist, existing file will be overwritten.""")
@click.option('--jobs',
              '-j',
              default=1,
              help='Number of parallel jobs (0 means all CPUs)')
@backend_param_option
@common_options
def run_eval(
        project_id,
        paths,
        limit,
        threshold,
        docs_limit,
        metric,
        metrics_file,
        results_file,
        jobs,
        backend_param):
    """
    Analyze documents and evaluate the result.

    Compare the results of automated indexing against a gold standard. The
    path may be either a TSV file with short documents or a directory with
    documents in separate files.
    """

    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param, project)

    import annif.eval
    eval_batch = annif.eval.EvaluationBatch(project.subjects)

    if results_file:
        try:
            print('', end='', file=results_file)
            click.echo('Writing per subject evaluation results to {!s}'.format(
                results_file.name))
        except Exception as e:
            raise NotSupportedException(
                "cannot open results-file for writing: " + str(e))
    docs = open_documents(paths, docs_limit)

    jobs, pool_class = annif.parallel.get_pool(jobs)

    project.initialize(parallel=True)
    psmap = annif.parallel.ProjectSuggestMap(
        project.registry, [project_id], backend_params, limit, threshold)

    with pool_class(jobs) as pool:
        for hits, uris, labels in pool.imap_unordered(
                psmap.suggest, docs.documents):
            eval_batch.evaluate(hits[project_id],
                                annif.corpus.SubjectSet((uris, labels)))

    template = "{0:<30}\t{1}"
    metrics = eval_batch.results(metrics=metric, results_file=results_file)
    for metric, score in metrics.items():
        click.echo(template.format(metric + ":", score))
    if metrics_file:
        json.dump(
            {metric_code(mname): val for mname, val in metrics.items()},
            metrics_file, indent=2)


@cli.command('optimize')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--docs-limit', '-d', default=None,
              type=click.IntRange(0, None),
              help='Maximum number of documents to use')
@backend_param_option
@common_options
def run_optimize(project_id, paths, docs_limit, backend_param):
    """
    Analyze documents, testing multiple limits and thresholds.

    Evaluate the analysis results for a directory with documents against a
    gold standard given in subject files. Test different limit/threshold
    values and report the precision, recall and F-measure of each combination
    of settings.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param, project)

    filter_batches = generate_filter_batches(project.subjects)

    ndocs = 0
    docs = open_documents(paths, docs_limit)
    for doc in docs.documents:
        raw_hits = project.suggest(doc.text, backend_params)
        hits = raw_hits.filter(project.subjects, limit=BATCH_MAX_LIMIT)
        assert isinstance(hits, ListSuggestionResult), \
            "Optimize should only be done with ListSuggestionResult " + \
            "as it would be very slow with VectorSuggestionResult."
        gold_subjects = annif.corpus.SubjectSet((doc.uris, doc.labels))
        for hit_filter, batch in filter_batches.values():
            batch.evaluate(hit_filter(hits), gold_subjects)
        ndocs += 1

    click.echo("\t".join(('Limit', 'Thresh.', 'Prec.', 'Rec.', 'F1')))

    best_scores = collections.defaultdict(float)
    best_params = {}

    template = "{:d}\t{:.02f}\t{:.04f}\t{:.04f}\t{:.04f}"
    # Store the batches in a list that gets consumed along the way
    # This way GC will have a chance to reclaim the memory
    filter_batches = list(filter_batches.items())
    while filter_batches:
        params, filter_batch = filter_batches.pop(0)
        metrics = ['Precision (doc avg)',
                   'Recall (doc avg)',
                   'F1 score (doc avg)']
        results = filter_batch[1].results(metrics=metrics)
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
    for metric in metrics:
        click.echo(
            template2.format(
                metric,
                best_scores[metric],
                best_params[metric][0],
                best_params[metric][1]))
    click.echo("Documents evaluated:\t{}".format(ndocs))


@cli.command('hyperopt')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--docs-limit', '-d', default=None,
              type=click.IntRange(0, None),
              help='Maximum number of documents to use')
@click.option('--trials', '-T', default=10, help='Number of trials')
@click.option('--jobs',
              '-j',
              default=1,
              help='Number of parallel runs (0 means all CPUs)')
@click.option('--metric', '-m', default='NDCG',
              help='Metric to optimize (default: NDCG)')
@click.option(
    '--results-file',
    '-r',
    type=click.File(
        'w',
        encoding='utf-8',
        errors='ignore',
        lazy=True),
    help="""Specify file path to write trial results as CSV.
    File directory must exist, existing file will be overwritten.""")
@common_options
def run_hyperopt(project_id, paths, docs_limit, trials, jobs, metric,
                 results_file):
    """
    Optimize the hyperparameters of a project using a validation corpus.
    """
    proj = get_project(project_id)
    documents = open_documents(paths, docs_limit)
    click.echo(f"Looking for optimal hyperparameters using {trials} trials")
    rec = proj.hyperopt(documents, trials, jobs, metric, results_file)
    click.echo(f"Got best {metric} score {rec.score:.4f} with:")
    click.echo("---")
    for line in rec.lines:
        click.echo(line)
    click.echo("---")


if __name__ == '__main__':
    cli()
