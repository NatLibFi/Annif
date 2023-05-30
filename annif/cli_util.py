"""Utility functions for Annif CLI commands"""
from __future__ import annotations

import collections
import itertools
import os
import sys
from typing import TYPE_CHECKING

import click
import click_log
from flask import current_app

import annif
from annif.exception import ConfigurationException
from annif.project import Access

if TYPE_CHECKING:
    from datetime import datetime
    from io import TextIOWrapper

    from click.core import Argument, Context, Option

    from annif.corpus.document import DocumentCorpus, DocumentList
    from annif.corpus.subject import SubjectIndex
    from annif.project import AnnifProject
    from annif.suggestion import SuggestionResult
    from annif.vocab import AnnifVocabulary

logger = annif.logger


def _set_project_config_file_path(
    ctx: Context, param: Option, value: str | None
) -> None:
    """Override the default path or the path given in env by CLI option"""
    with ctx.obj.load_app().app_context():
        if value:
            current_app.config["PROJECTS_CONFIG_PATH"] = value


def common_options(f):
    """Decorator to add common options for all CLI commands"""
    f = click.option(
        "-p",
        "--projects",
        help="Set path to project configuration file or directory",
        type=click.Path(dir_okay=True, exists=True),
        callback=_set_project_config_file_path,
        expose_value=False,
        is_eager=True,
    )(f)
    return click_log.simple_verbosity_option(logger)(f)


def project_id(f):
    """Decorator to add a project ID parameter to a CLI command"""
    return click.argument("project_id", shell_complete=complete_param)(f)


def backend_param_option(f):
    """Decorator to add an option for CLI commands to override BE parameters"""
    return click.option(
        "--backend-param",
        "-b",
        multiple=True,
        help="Override backend parameter of the config file. "
        + "Syntax: `-b <backend>.<parameter>=<value>`.",
    )(f)


def docs_limit_option(f):
    """Decorator to add an option for CLI commands to limit the number of documents to
    use"""
    return click.option(
        "--docs-limit",
        "-d",
        default=None,
        type=click.IntRange(0, None),
        help="Maximum number of documents to use",
    )(f)


def get_project(project_id: str) -> AnnifProject:
    """
    Helper function to get a project by ID and bail out if it doesn't exist"""
    try:
        return annif.registry.get_project(project_id, min_access=Access.private)
    except ValueError:
        click.echo("No projects found with id '{0}'.".format(project_id), err=True)
        sys.exit(1)


def get_vocab(vocab_id: str) -> AnnifVocabulary:
    """
    Helper function to get a vocabulary by ID and bail out if it doesn't
    exist"""
    try:
        return annif.registry.get_vocab(vocab_id, min_access=Access.private)
    except ValueError:
        click.echo(f"No vocabularies found with the id '{vocab_id}'.", err=True)
        sys.exit(1)


def make_list_template(*rows) -> str:
    """Helper function to create a template for a list of entries with fields of
    variable width. The width of each field is determined by the longest item in the
    field in the given rows."""

    max_field_widths = collections.defaultdict(int)
    for row in rows:
        for field_ind, item in enumerate(row):
            max_field_widths[field_ind] = max(max_field_widths[field_ind], len(item))

    return "  ".join(
        [
            f"{{{field_ind}: <{field_width}}}"
            for field_ind, field_width in max_field_widths.items()
        ]
    )


def format_datetime(dt: datetime | None) -> str:
    """Helper function to format a datetime object as a string in the local time."""
    if dt is None:
        return "-"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def open_documents(
    paths: tuple[str, ...],
    subject_index: SubjectIndex,
    vocab_lang: str,
    docs_limit: int | None,
) -> DocumentCorpus:
    """Helper function to open a document corpus from a list of pathnames,
    each of which is either a TSV file or a directory of TXT files. For
    directories with subjects in TSV files, the given vocabulary language
    will be used to convert subject labels into URIs. The corpus will be
    returned as an instance of DocumentCorpus or LimitingDocumentCorpus."""

    def open_doc_path(path, subject_index):
        """open a single path and return it as a DocumentCorpus"""
        if os.path.isdir(path):
            return annif.corpus.DocumentDirectory(
                path, subject_index, vocab_lang, require_subjects=True
            )
        return annif.corpus.DocumentFile(path, subject_index)

    if len(paths) == 0:
        logger.warning("Reading empty file")
        docs = open_doc_path(os.path.devnull, subject_index)
    elif len(paths) == 1:
        docs = open_doc_path(paths[0], subject_index)
    else:
        corpora = [open_doc_path(path, subject_index) for path in paths]
        docs = annif.corpus.CombinedCorpus(corpora)
    if docs_limit is not None:
        docs = annif.corpus.LimitingDocumentCorpus(docs, docs_limit)
    return docs


def open_text_documents(paths: tuple[str, ...], docs_limit: int | None) -> DocumentList:
    """
    Helper function to read text documents from the given file paths. Returns a
    DocumentList object with Documents having no subjects. If a path is "-", the
    document text is read from standard input. The maximum number of documents to read
    is set by docs_limit parameter.
    """

    def _docs(paths):
        for path in paths:
            if path == "-":
                doc = annif.corpus.Document(text=sys.stdin.read(), subject_set=None)
            else:
                with open(path, errors="replace", encoding="utf-8-sig") as docfile:
                    doc = annif.corpus.Document(text=docfile.read(), subject_set=None)
            yield doc

    return annif.corpus.DocumentList(_docs(paths[:docs_limit]))


def show_hits(
    hits: SuggestionResult,
    project: AnnifProject,
    lang: str,
    file: TextIOWrapper | None = None,
) -> None:
    """
    Print subject suggestions to the console or a file. The suggestions are displayed as
    a table, with one row per hit. Each row contains the URI, label, possible notation,
    and score of the suggestion. The label is given in the specified language.
    """
    template = "<{}>\t{}\t{:.04f}"
    for hit in hits:
        subj = project.subjects[hit.subject_id]
        line = template.format(
            subj.uri,
            "\t".join(filter(None, (subj.labels[lang], subj.notation))),
            hit.score,
        )
        click.echo(line, file=file)


def parse_backend_params(
    backend_param: tuple[str, ...] | tuple[()], project: AnnifProject
) -> collections.defaultdict[str, dict[str, str]]:
    """Parse a list of backend parameters given with the --backend-param
    option into a nested dict structure"""
    backend_params = collections.defaultdict(dict)
    for beparam in backend_param:
        backend, param = beparam.split(".", 1)
        key, val = param.split("=", 1)
        _validate_backend_params(backend, beparam, project)
        backend_params[backend][key] = val
    return backend_params


def _validate_backend_params(backend: str, beparam: str, project: AnnifProject) -> None:
    if backend != project.config["backend"]:
        raise ConfigurationException(
            'The backend {} in CLI option "-b {}" not matching the project'
            " backend {}.".format(backend, beparam, project.config["backend"])
        )


def generate_filter_params(filter_batch_max_limit: int) -> list[tuple[int, float]]:
    limits = range(1, filter_batch_max_limit + 1)
    thresholds = [i * 0.05 for i in range(20)]
    return list(itertools.product(limits, thresholds))


def _get_completion_choices(
    param: Argument,
) -> dict[str, AnnifVocabulary] | dict[str, AnnifProject] | list:
    if param.name == "project_id":
        return annif.registry.get_projects()
    elif param.name == "vocab_id":
        return annif.registry.get_vocabs()
    else:
        return []


def complete_param(ctx: Context, param: Argument, incomplete: str) -> list[str]:
    with ctx.obj.load_app().app_context():
        return [
            choice
            for choice in _get_completion_choices(param)
            if choice.startswith(incomplete)
        ]
