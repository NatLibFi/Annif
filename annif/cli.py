"""Definitions for command-line (Click) commands for invoking Annif
operations and printing the results to console."""


import collections
import importlib
import json
import os.path
import re
import sys

import click
import click_log
from flask.cli import FlaskGroup

import annif
import annif.corpus
import annif.parallel
import annif.project
import annif.registry
from annif import cli_util
from annif.exception import NotInitializedException, NotSupportedException
from annif.project import Access
from annif.util import metric_code

logger = annif.logger
click_log.basic_config(logger)


if len(sys.argv) > 1 and sys.argv[1] in ("run", "routes"):
    create_app = annif.create_app  # Use Flask with Connexion
else:
    # Connexion is not needed for most CLI commands, use plain Flask
    create_app = annif.create_flask_app

cli = FlaskGroup(create_app=create_app, add_version_option=False)
cli = click.version_option(message="%(version)s")(cli)


@cli.command("list-projects")
@cli_util.common_options
@click_log.simple_verbosity_option(logger, default="ERROR")
def run_list_projects():
    """
    List available projects.
    \f
    Show a list of currently defined projects. Projects are defined in a
    configuration file, normally called ``projects.cfg``. See `Project
    configuration
    <https://github.com/NatLibFi/Annif/wiki/Project-configuration>`_
    for details.
    """

    column_headings = (
        "Project ID",
        "Project Name",
        "Vocabulary ID",
        "Language",
        "Trained",
        "Modification time",
    )
    table = [
        (
            proj.project_id,
            proj.name,
            proj.vocab.vocab_id if proj.vocab_spec else "-",
            proj.language,
            str(proj.is_trained),
            cli_util.format_datetime(proj.modification_time),
        )
        for proj in annif.registry.get_projects(min_access=Access.private).values()
    ]
    template = cli_util.make_list_template(column_headings, *table)
    header = template.format(*column_headings)
    click.echo(header)
    click.echo("-" * len(header))
    for row in table:
        click.echo(template.format(*row))


@cli.command("show-project")
@cli_util.project_id
@cli_util.common_options
def run_show_project(project_id):
    """
    Show information about a project.
    """

    proj = cli_util.get_project(project_id)
    click.echo(f"Project ID:        {proj.project_id}")
    click.echo(f"Project Name:      {proj.name}")
    click.echo(f"Language:          {proj.language}")
    click.echo(f"Vocabulary:        {proj.vocab.vocab_id}")
    click.echo(f"Vocab language:    {proj.vocab_lang}")
    click.echo(f"Access:            {proj.access.name}")
    click.echo(f"Backend:           {proj.backend.name}")
    click.echo(f"Trained:           {proj.is_trained}")
    click.echo(f"Modification time: {cli_util.format_datetime(proj.modification_time)}")


@cli.command("clear")
@cli_util.project_id
@cli_util.common_options
def run_clear_project(project_id):
    """
    Initialize the project to its original, untrained state.
    """
    proj = cli_util.get_project(project_id)
    proj.remove_model_data()


@cli.command("list-vocabs")
@cli_util.common_options
@click_log.simple_verbosity_option(logger, default="ERROR")
def run_list_vocabs():
    """
    List available vocabularies.
    """

    column_headings = ("Vocabulary ID", "Languages", "Size", "Loaded")
    table = []
    for vocab in annif.registry.get_vocabs(min_access=Access.private).values():
        try:
            languages = ",".join(sorted(vocab.languages))
            size = len(vocab)
            loaded = True
        except NotInitializedException:
            languages = "-"
            size = "-"
            loaded = False
        row = (vocab.vocab_id, languages, str(size), str(loaded))
        table.append(row)

    template = cli_util.make_list_template(column_headings, *table)
    header = template.format(*column_headings)
    click.echo(header)
    click.echo("-" * len(header))
    for row in table:
        click.echo(template.format(*row))


@cli.command("load-vocab")
@click.argument("vocab_id", shell_complete=cli_util.complete_param)
@click.argument("subjectfile", type=click.Path(exists=True, dir_okay=False))
@click.option("--language", "-L", help="Language of subject file")
@click.option(
    "--force",
    "-f",
    default=False,
    is_flag=True,
    help="Replace existing vocabulary completely instead of updating it",
)
@cli_util.common_options
def run_load_vocab(vocab_id, language, force, subjectfile):
    """
    Load a vocabulary from a subject file.
    """
    vocab = cli_util.get_vocab(vocab_id)
    if annif.corpus.SubjectFileSKOS.is_rdf_file(subjectfile):
        # SKOS/RDF file supported by rdflib
        subjects = annif.corpus.SubjectFileSKOS(subjectfile)
        click.echo(f"Loading vocabulary from SKOS file {subjectfile}...")
    elif annif.corpus.SubjectFileCSV.is_csv_file(subjectfile):
        # CSV file
        subjects = annif.corpus.SubjectFileCSV(subjectfile)
        click.echo(f"Loading vocabulary from CSV file {subjectfile}...")
    else:
        # probably a TSV file - we need to know its language
        if not language:
            click.echo(
                "Please use --language option to set the language of a TSV vocabulary.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Loading vocabulary from TSV file {subjectfile}...")
        subjects = annif.corpus.SubjectFileTSV(subjectfile, language)
    vocab.load_vocabulary(subjects, force=force)


@cli.command("train")
@cli_util.project_id
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--cached/--no-cached",
    "-c/-C",
    default=False,
    help="Reuse preprocessed training data from previous run",
)
@click.option(
    "--jobs",
    "-j",
    default=0,
    help="Number of parallel jobs (0 means choose automatically)",
)
@cli_util.docs_limit_option
@cli_util.backend_param_option
@cli_util.common_options
def run_train(project_id, paths, cached, docs_limit, jobs, backend_param):
    """
    Train a project on a collection of documents.
    \f
    This will train the project using the documents from ``PATHS`` (directories
    or possibly gzipped TSV files) in a single batch operation. If ``--cached``
    is set, preprocessed training data from the previous run is reused instead
    of documents input; see `Reusing preprocessed training data
    <https://github.com/NatLibFi/Annif/wiki/
    Reusing-preprocessed-training-data>`_.
    """
    proj = cli_util.get_project(project_id)
    backend_params = cli_util.parse_backend_params(backend_param, proj)
    if cached:
        if len(paths) > 0:
            raise click.UsageError(
                "Corpus paths cannot be given when using --cached option."
            )
        documents = "cached"
    else:
        documents = cli_util.open_documents(
            paths, proj.subjects, proj.vocab_lang, docs_limit
        )
    proj.train(documents, backend_params, jobs)


@cli.command("learn")
@cli_util.project_id
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@cli_util.docs_limit_option
@cli_util.backend_param_option
@cli_util.common_options
def run_learn(project_id, paths, docs_limit, backend_param):
    """
    Further train an existing project on a collection of documents.
    \f
    Similar to the ``train`` command. This will continue training an already
    trained project using the documents given by ``PATHS`` in a single batch
    operation. Not supported by all backends.
    """
    proj = cli_util.get_project(project_id)
    backend_params = cli_util.parse_backend_params(backend_param, proj)
    documents = cli_util.open_documents(
        paths, proj.subjects, proj.vocab_lang, docs_limit
    )
    proj.learn(documents, backend_params)


@cli.command("suggest")
@cli_util.project_id
@click.argument(
    "paths", type=click.Path(dir_okay=False, exists=True, allow_dash=True), nargs=-1
)
@click.option("--limit", "-l", default=10, help="Maximum number of subjects")
@click.option("--threshold", "-t", default=0.0, help="Minimum score threshold")
@click.option("--language", "-L", help="Language of subject labels")
@cli_util.docs_limit_option
@cli_util.backend_param_option
@cli_util.common_options
def run_suggest(
    project_id, paths, limit, threshold, language, backend_param, docs_limit
):
    """
    Suggest subjects for a single document from standard input or for one or more
    document file(s) given its/their path(s).
    \f
    This will read a text document from standard input and suggest subjects for
    it, or if given path(s) to file(s), suggest subjects for it/them.
    """
    project = cli_util.get_project(project_id)
    lang = language or project.vocab_lang
    if lang not in project.vocab.languages:
        raise click.BadParameter(f'language "{lang}" not supported by vocabulary')
    backend_params = cli_util.parse_backend_params(backend_param, project)

    if paths and not (len(paths) == 1 and paths[0] == "-"):
        docs = cli_util.open_text_documents(paths, docs_limit)
        results = project.suggest_corpus(docs, backend_params).filter(limit, threshold)
        for (
            suggestions,
            path,
        ) in zip(results, paths):
            click.echo(f"Suggestions for {path}")
            cli_util.show_hits(suggestions, project, lang)
    else:
        text = sys.stdin.read()
        suggestions = project.suggest([text], backend_params).filter(limit, threshold)[
            0
        ]
        cli_util.show_hits(suggestions, project, lang)


@cli.command("index")
@cli_util.project_id
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--suffix", "-s", default=".annif", help="File name suffix for result files"
)
@click.option(
    "--force/--no-force",
    "-f/-F",
    default=False,
    help="Force overwriting of existing result files",
)
@click.option("--limit", "-l", default=10, help="Maximum number of subjects")
@click.option("--threshold", "-t", default=0.0, help="Minimum score threshold")
@click.option("--language", "-L", help="Language of subject labels")
@cli_util.backend_param_option
@cli_util.common_options
def run_index(
    project_id, directory, suffix, force, limit, threshold, language, backend_param
):
    """
    Index a directory with documents, suggesting subjects for each document.
    Write the results in TSV files with the given suffix (``.annif`` by
    default).
    """
    project = cli_util.get_project(project_id)
    lang = language or project.vocab_lang
    if lang not in project.vocab.languages:
        raise click.BadParameter(f'language "{lang}" not supported by vocabulary')
    backend_params = cli_util.parse_backend_params(backend_param, project)

    documents = annif.corpus.DocumentDirectory(directory, require_subjects=False)
    results = project.suggest_corpus(documents, backend_params).filter(limit, threshold)

    for (docfilename, _), suggestions in zip(documents, results):
        subjectfilename = re.sub(r"\.txt$", suffix, docfilename)
        if os.path.exists(subjectfilename) and not force:
            click.echo(
                "Not overwriting {} (use --force to override)".format(subjectfilename)
            )
            continue
        with open(subjectfilename, "w", encoding="utf-8") as subjfile:
            cli_util.show_hits(suggestions, project, lang, file=subjfile)


@cli.command("eval")
@cli_util.project_id
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option("--limit", "-l", default=10, help="Maximum number of subjects")
@click.option("--threshold", "-t", default=0.0, help="Minimum score threshold")
@click.option(
    "--metric",
    "-m",
    default=[],
    multiple=True,
    help="Metric to calculate (default: all)",
)
@click.option(
    "--metrics-file",
    "-M",
    type=click.File("w", encoding="utf-8", errors="ignore", lazy=True),
    help="""Specify file in order to write evaluation metrics in JSON format.
    File directory must exist, existing file will be overwritten.""",
)
@click.option(
    "--results-file",
    "-r",
    type=click.File("w", encoding="utf-8", errors="ignore", lazy=True),
    help="""Specify file in order to write non-aggregated results per subject.
    File directory must exist, existing file will be overwritten.""",
)
@click.option(
    "--jobs", "-j", default=1, help="Number of parallel jobs (0 means all CPUs)"
)
@cli_util.docs_limit_option
@cli_util.backend_param_option
@cli_util.common_options
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
    backend_param,
):
    """
    Suggest subjects for documents and evaluate the results by comparing
    against a gold standard.
    \f
    With this command the documents from ``PATHS`` (directories or possibly
    gzipped TSV files) will be assigned subject suggestions and then
    statistical measures are calculated that quantify how well the suggested
    subjects match the gold-standard subjects in the documents.

    Normally the output is the list of the metrics calculated across documents.
    If ``--results-file <FILENAME>`` option is given, the metrics are
    calculated separately for each subject, and written to the given file.
    """

    project = cli_util.get_project(project_id)
    backend_params = cli_util.parse_backend_params(backend_param, project)

    import annif.eval

    eval_batch = annif.eval.EvaluationBatch(project.subjects)

    if results_file:
        try:
            print("", end="", file=results_file)
            click.echo(
                "Writing per subject evaluation results to {!s}".format(
                    results_file.name
                )
            )
        except Exception as e:
            raise NotSupportedException(
                "cannot open results-file for writing: " + str(e)
            )
    corpus = cli_util.open_documents(
        paths, project.subjects, project.vocab_lang, docs_limit
    )
    jobs, pool_class = annif.parallel.get_pool(jobs)

    project.initialize(parallel=True)
    psmap = annif.parallel.ProjectSuggestMap(
        project.registry, [project_id], backend_params, limit, threshold
    )

    with pool_class(jobs) as pool:
        for hit_sets, subject_sets in pool.imap_unordered(
            psmap.suggest_batch, corpus.doc_batches
        ):
            eval_batch.evaluate_many(hit_sets[project_id], subject_sets)

    template = "{0:<30}\t{1:{fmt_spec}}"
    metrics = eval_batch.results(
        metrics=metric, results_file=results_file, language=project.vocab_lang
    )
    for metric, score in metrics.items():
        if isinstance(score, int):
            fmt_spec = "d"
        elif isinstance(score, float):
            fmt_spec = ".04f"
        click.echo(template.format(metric + ":", score, fmt_spec=fmt_spec))
    if metrics_file:
        json.dump(
            {metric_code(mname): val for mname, val in metrics.items()},
            metrics_file,
            indent=2,
        )


FILTER_BATCH_MAX_LIMIT = 15
OPTIMIZE_METRICS = ["Precision (doc avg)", "Recall (doc avg)", "F1 score (doc avg)"]


@cli.command("optimize")
@cli_util.project_id
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--jobs", "-j", default=1, help="Number of parallel jobs (0 means all CPUs)"
)
@cli_util.docs_limit_option
@cli_util.backend_param_option
@cli_util.common_options
def run_optimize(project_id, paths, jobs, docs_limit, backend_param):
    """
    Suggest subjects for documents, testing multiple limits and thresholds.
    \f
    This command will use different limit (maximum number of subjects) and
    score threshold values when assigning subjects to each document given by
    ``PATHS`` and compare the results against the gold standard subjects in the
    documents. The output is a list of parameter combinations and their scores.
    From the output, you can determine the optimum limit and threshold
    parameters depending on which measure you want to target.
    """
    project = cli_util.get_project(project_id)
    backend_params = cli_util.parse_backend_params(backend_param, project)
    filter_params = cli_util.generate_filter_params(FILTER_BATCH_MAX_LIMIT)

    import annif.eval

    corpus = cli_util.open_documents(
        paths, project.subjects, project.vocab_lang, docs_limit
    )

    jobs, pool_class = annif.parallel.get_pool(jobs)

    project.initialize(parallel=True)
    psmap = annif.parallel.ProjectSuggestMap(
        project.registry,
        [project_id],
        backend_params,
        limit=FILTER_BATCH_MAX_LIMIT,
        threshold=0.0,
    )

    ndocs = 0
    suggestion_batches = []
    subject_set_batches = []
    with pool_class(jobs) as pool:
        for suggestion_batch, subject_sets in pool.imap_unordered(
            psmap.suggest_batch, corpus.doc_batches
        ):
            ndocs += len(suggestion_batch[project_id])
            suggestion_batches.append(suggestion_batch[project_id])
            subject_set_batches.append(subject_sets)

    from annif.suggestion import SuggestionResults

    orig_suggestion_results = SuggestionResults(suggestion_batches)

    click.echo("\t".join(("Limit", "Thresh.", "Prec.", "Rec.", "F1")))

    best_scores = collections.defaultdict(float)
    best_params = {}

    template = "{:d}\t{:.02f}\t{:.04f}\t{:.04f}\t{:.04f}"
    import annif.eval

    for limit, threshold in filter_params:
        eval_batch = annif.eval.EvaluationBatch(project.subjects)
        filtered_results = orig_suggestion_results.filter(limit, threshold)
        for batch, subject_sets in zip(filtered_results.batches, subject_set_batches):
            eval_batch.evaluate_many(batch, subject_sets)
        results = eval_batch.results(metrics=OPTIMIZE_METRICS)
        for metric, score in results.items():
            if score >= best_scores[metric]:
                best_scores[metric] = score
                best_params[metric] = (limit, threshold)
        click.echo(
            template.format(
                limit,
                threshold,
                results["Precision (doc avg)"],
                results["Recall (doc avg)"],
                results["F1 score (doc avg)"],
            )
        )

    click.echo()
    template2 = "Best {:>19}: {:.04f}\tLimit: {:d}\tThreshold: {:.02f}"
    for metric in OPTIMIZE_METRICS:
        click.echo(
            template2.format(
                metric,
                best_scores[metric],
                best_params[metric][0],
                best_params[metric][1],
            )
        )
    click.echo("Documents evaluated:\t{}".format(ndocs))


@cli.command("hyperopt")
@cli_util.project_id
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option("--trials", "-T", default=10, help="Number of trials")
@click.option(
    "--jobs", "-j", default=1, help="Number of parallel runs (0 means all CPUs)"
)
@click.option(
    "--metric", "-m", default="NDCG", help="Metric to optimize (default: NDCG)"
)
@click.option(
    "--results-file",
    "-r",
    type=click.File("w", encoding="utf-8", errors="ignore", lazy=True),
    help="""Specify file path to write trial results as CSV.
    File directory must exist, existing file will be overwritten.""",
)
@cli_util.docs_limit_option
@cli_util.common_options
def run_hyperopt(project_id, paths, docs_limit, trials, jobs, metric, results_file):
    """
    Optimize the hyperparameters of a project using validation documents from
    ``PATHS``. Not supported by all backends. Output is a list of trial results
    and a report of the best performing parameters.
    """
    proj = cli_util.get_project(project_id)
    documents = cli_util.open_documents(
        paths, proj.subjects, proj.vocab_lang, docs_limit
    )
    click.echo(f"Looking for optimal hyperparameters using {trials} trials")
    rec = proj.hyperopt(documents, trials, jobs, metric, results_file)
    click.echo(f"Got best {metric} score {rec.score:.4f} with:")
    click.echo("---")
    for line in rec.lines:
        click.echo(line)
    click.echo("---")


@cli.command("completion")
@click.option("--bash", "shell", flag_value="bash")
@click.option("--zsh", "shell", flag_value="zsh")
@click.option("--fish", "shell", flag_value="fish")
def completion(shell):
    """Generate the script for tab-key autocompletion for the given shell. To enable the
    completion support in your current bash terminal session run\n
        source <(annif completion --bash)
    """

    if shell is None:
        raise click.UsageError("Shell not given, try --bash, --zsh or --fish")

    script = os.popen(f"_ANNIF_COMPLETE={shell}_source annif").read()
    click.echo(f"# Generated by Annif {importlib.metadata.version('annif')}")
    click.echo(script)


if __name__ == "__main__":
    cli()
