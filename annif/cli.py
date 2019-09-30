"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import os.path
import re
import sys
import click
import click_log
from flask import current_app
from flask.cli import FlaskGroup, ScriptInfo
import annif
import annif.corpus
import annif.eval
import annif.project
from annif.project import Access
from annif.suggestion import SuggestionFilter

logger = annif.logger
click_log.basic_config(logger)

cli = FlaskGroup(create_app=annif.create_app)


def get_project(project_id):
    """
    Helper function to get a project by ID and bail out if it doesn't exist"""
    try:
        return annif.project.get_project(project_id, min_access=Access.hidden)
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

    if len(paths) == 0:
        logger.warning('Reading empty file')
        docs = open_doc_path(os.path.devnull)
    elif len(paths) == 1:
        docs = open_doc_path(paths[0])
    else:
        corpora = [open_doc_path(path) for path in paths]
        docs = annif.corpus.CombinedCorpus(corpora)
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
            hit_filter = SuggestionFilter(limit, threshold)
            batch = annif.eval.EvaluationBatch(subjects)
            filter_batches[(limit, threshold)] = (hit_filter, batch)
    return filter_batches


def set_project_config_file_path(ctx, param, value):
    """Override the default path or the path given in env by CLI option"""
    with ctx.ensure_object(ScriptInfo).load_app().app_context():
        if value:
            current_app.config['PROJECTS_FILE'] = value


def common_options(f):
    """Decorator to add common options for all CLI commands"""
    f = click.option(
        '-p', '--projects', help='Set path to projects.cfg',
        type=click.Path(dir_okay=False, exists=True),
        callback=set_project_config_file_path, expose_value=False,
        is_eager=True)(f)
    f = click_log.simple_verbosity_option(logger)(f)
    return f


@cli.command('list-projects')
@common_options
def run_list_projects():
    """
    List available projects.
    """

    template = "{0: <25}{1: <45}{2: <8}"
    header = template.format("Project ID", "Project Name", "Language")
    click.echo(header)
    click.echo("-" * len(header))
    for proj in annif.project.get_projects(min_access=Access.private).values():
        click.echo(template.format(proj.project_id, proj.name, proj.language))


@cli.command('show-project')
@click.argument('project_id')
@common_options
def run_show_project(project_id):
    """
    Show information about a project.
    """

    proj = get_project(project_id)
    template = "{0:<20}{1}"
    click.echo(template.format('Project ID:', proj.project_id))
    click.echo(template.format('Project Name:', proj.name))
    click.echo(template.format('Language:', proj.language))
    click.echo(template.format('Access:', proj.access.name))


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
@common_options
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
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@common_options
def run_train(project_id, paths):
    """
    Train a project on a collection of documents.
    """
    proj = get_project(project_id)
    documents = open_documents(paths)
    proj.train(documents)


@cli.command('learn')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@common_options
def run_learn(project_id, paths):
    """
    Further train an existing project on a collection of documents.
    """
    proj = get_project(project_id)
    documents = open_documents(paths)
    proj.learn(documents)


@cli.command('suggest')
@click.argument('project_id')
@click.option('--limit', default=10, help='Maximum number of subjects')
@click.option('--threshold', default=0.0, help='Minimum score threshold')
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
@common_options
def run_suggest(project_id, limit, threshold, backend_param):
    """
    Suggest subjects for a single document from standard input.
    """
    project = get_project(project_id)
    text = sys.stdin.read()
    backend_params = parse_backend_params(backend_param)
    hit_filter = SuggestionFilter(limit, threshold)
    hits = hit_filter(project.suggest(text, backend_params))
    for hit in hits:
        click.echo("<{}>\t{}\t{}".format(hit.uri, hit.label, hit.score))


@cli.command('index')
@click.argument('project_id')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
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
@common_options
def run_index(project_id, directory, suffix, force,
              limit, threshold, backend_param):
    """
    Index a directory with documents, suggesting subjects for each document.
    Write the results in TSV files with the given suffix.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)
    hit_filter = SuggestionFilter(limit, threshold)

    for docfilename, dummy_subjectfn in annif.corpus.DocumentDirectory(
            directory, require_subjects=False):
        with open(docfilename, encoding='utf-8') as docfile:
            text = docfile.read()
        subjectfilename = re.sub(r'\.txt$', suffix, docfilename)
        if os.path.exists(subjectfilename) and not force:
            click.echo(
                "Not overwriting {} (use --force to override)".format(
                    subjectfilename))
            continue
        with open(subjectfilename, 'w', encoding='utf-8') as subjfile:
            results = project.suggest(text, backend_params)
            for hit in hit_filter(results):
                line = "<{}>\t{}\t{}".format(hit.uri, hit.label, hit.score)
                click.echo(line, file=subjfile)


@cli.command('eval')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--limit', default=10, help='Maximum number of subjects')
@click.option('--threshold', default=0.0, help='Minimum score threshold')
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
@common_options
def run_eval(project_id, paths, limit, threshold, backend_param):
    """
    Analyze documents and evaluate the result.

    Compare the results of automated indexing against a gold standard. The
    path may be either a TSV file with short documents or a directory with
    documents in separate files.
    """
    project = get_project(project_id)
    backend_params = parse_backend_params(backend_param)

    hit_filter = SuggestionFilter(limit=limit, threshold=threshold)
    eval_batch = annif.eval.EvaluationBatch(project.subjects)

    docs = open_documents(paths)
    for doc in docs.documents:
        results = project.suggest(doc.text, backend_params)
        hits = hit_filter(results)
        eval_batch.evaluate(hits,
                            annif.corpus.SubjectSet((doc.uris, doc.labels)))

    template = "{0:<20}\t{1}"
    for metric, score in eval_batch.results().items():
        click.echo(template.format(metric + ":", score))


@cli.command('learning-curves')
@click.argument('project_id')
@click.argument('train-paths', type=click.Path(), nargs=-1)
@click.option('--num-points', '-n', default=5, help='TODO')  # TODO
@click.option('--test-frac', '-f', default=0.2,
              help='The fraction of documents to use as a test set.')
@click.option('--eval-train/--no-eval-train', default=True,
              help='Evaluate scores also for the training set.')
@click.option('--eval-metrics', default='simple', help='Evaluate either ')  # TODO
@click.option('--limit', default=10, help='Maximum number of subjects')
@click.option('--threshold', default=0.0, help='Minimum score threshold')
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
@common_options
def run_learning_curves(project_id, train_paths, num_points, test_frac,
                        eval_train, eval_metrics, limit, threshold,
                        backend_param):
    """ # TODO
    """
    backend_params = parse_backend_params(backend_param)
    project = get_project(project_id)
    # TODO tmp datadir?
    # project._datadir_path = 'tmp_data'
    from annif.corpus import DocumentList
    from itertools import islice
    import warnings
    from more_itertools import ilen

    def _evaluate(docs, logfile):
        eval_batch = annif.eval.EvaluationBatch(project.subjects)
        hit_filter = SuggestionFilter(limit=limit, threshold=threshold)
        for doc in docs.documents:
            results = project.suggest(doc.text, backend_params)
            hits = hit_filter(results)
            eval_batch.evaluate(
                hits, annif.corpus.SubjectSet((doc.uris, doc.labels)))
        results = eval_batch.results(metrics=eval_metrics)
        if ind == 1:
            with open(logfile, 'w') as lgfile:
                line = 'Documents in training\t' + '\t'.join(results.keys())
                print(line, file=lgfile)
        with open(logfile, 'a') as lgfile:
            line = str(num_docs_train_part) + '\t' + '\t'.join(
                (str(v) for v in results.values()))
            print(line, file=lgfile)

    def open_docs_slice(train_paths, start, stop):
        docs = open_documents(train_paths)
        return DocumentList(islice(docs.documents, start, stop))

    docs = open_documents(train_paths)
    num_docs = ilen(docs.documents)
    num_docs_test = round(test_frac * num_docs)
    num_docs_train = num_docs - num_docs_test
    click.echo('Documents total: {}, in training set: {}, in test set: {}'
               .format(num_docs, num_docs_train, num_docs_test))

    for ind in range(1, num_points+1):
        num_docs_train_part = round(ind / num_points * num_docs_train)
#        base = 2
#        num_docs_train_part = round(
#            base ** ind / base ** num_points * num_docs_train)
        click.echo('Point {}/{}, documents in partial training set: {}'.format(
                ind, num_points, num_docs_train_part))
#        with warnings.catch_warnings():
#            warnings.simplefilter('ignore')
        docs_train = open_docs_slice(train_paths, num_docs_test,
                                     num_docs_test + num_docs_train_part)
        project.train(docs_train)

        if eval_train:
            click.echo('Evaluating on train set')
            docs_train = open_docs_slice(train_paths, num_docs_test,
                                         num_docs_test + num_docs_train_part)
            _evaluate(docs_train,
                      'logfile-' + project.project_id + '-train.tsv')
        click.echo('Evaluating on test set')
        docs_test = open_docs_slice(train_paths, 0, num_docs_test)
        _evaluate(docs_test, 'logfile-' + project.project_id + '-test.tsv')


@cli.command('optimize')
@click.argument('project_id')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--backend-param', '-b', multiple=True,
              help='Backend parameters to override')
@common_options
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
        hits = project.suggest(doc.text, backend_params)
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
        results = filter_batch[1].results(metrics='simple')
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
