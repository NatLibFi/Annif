"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the OpenAPI specification."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import connexion

import annif.registry
import annif.simplemma_util
from annif.corpus import Document, DocumentList, SubjectSet
from annif.exception import AnnifException, NotEnabledException
from annif.project import Access
from annif.util import suggestion_results_to_list

if TYPE_CHECKING:
    from connexion.lifecycle import ConnexionResponse

    from annif.corpus.subject import SubjectIndex


def project_not_found_error(project_id: str) -> ConnexionResponse:
    """return a Connexion error object when a project is not found"""

    return connexion.problem(
        status=404,
        title="Project not found",
        detail="Project '{}' not found".format(project_id),
    )


def learning_not_enabled_error(project_id) -> ConnexionResponse:
    """return a Connexion error object when a project is not configured for learning"""

    return connexion.problem(
        status=403,
        title="Learning not allowed",
        detail=f"Project '{project_id}' is not configured to allow learning via API",
    )


def server_error(
    err: AnnifException,
) -> ConnexionResponse:
    """return a Connexion error object when there is a server error (project
    or backend problem)"""

    return connexion.problem(
        status=503, title="Service unavailable", detail=err.format_message()
    )


def show_info() -> tuple:
    """return version of annif and a title for the api according to OpenAPI spec"""

    result = {"title": "Annif REST API", "version": importlib.metadata.version("annif")}
    return result, 200, {"Content-Type": "application/json"}


def language_not_supported_error(lang: str) -> ConnexionResponse:
    """return a Connexion error object when attempting to use unsupported language"""

    return connexion.problem(
        status=400,
        title="Bad Request",
        detail=f'language "{lang}" not supported by vocabulary',
    )


def list_vocabs() -> tuple:
    """return a dict with vocabularies formatted according to OpenAPI spec"""

    result = {
        "vocabs": [
            vocab.dump()
            for vocab in annif.registry.get_vocabs(min_access=Access.public).values()
        ]
    }
    return result, 200, {"Content-Type": "application/json"}


def list_projects() -> tuple:
    """return a dict with projects formatted according to OpenAPI spec"""

    result = {
        "projects": [
            proj.dump()
            for proj in annif.registry.get_projects(min_access=Access.public).values()
        ]
    }
    return result, 200, {"Content-Type": "application/json"}


def show_project(
    project_id: str,
) -> dict | ConnexionResponse:
    """return a single project formatted according to OpenAPI spec"""

    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)
    return project.dump(), 200, {"Content-Type": "application/json"}


def detect_language(body: dict[str, Any]):
    """return scores for detected languages formatted according to Swagger spec"""

    text = body.get("text")
    languages = body.get("languages")

    try:
        proportions = annif.simplemma_util.detect_language(text, tuple(languages))
    except ValueError:
        return connexion.problem(
            status=400,
            title="Bad Request",
            detail="unsupported candidate languages",
        )

    result = {
        "results": [
            {"language": lang if lang != "unk" else None, "score": score}
            for lang, score in proportions.items()
        ]
    }
    return result, 200, {"Content-Type": "application/json"}


def _is_error(result: list[dict[str, list]] | ConnexionResponse) -> bool:
    return (
        isinstance(result, connexion.lifecycle.ConnexionResponse)
        and result.status_code >= 400
    )


def suggest(
    project_id: str, body: dict[str, Any]
) -> dict[str, list] | ConnexionResponse:
    """suggest subjects for the given text and return a dict with results
    formatted according to OpenAPI spec"""

    parameters = dict(
        (key, body[key]) for key in ["language", "limit", "threshold"] if key in body
    )
    metadata = {
        key[len("metadata_") :]: value
        for key, value in body.items()
        if key.startswith("metadata_")
    }
    documents = [{"text": body["text"], "metadata": metadata}]
    result = _suggest(project_id, documents, parameters)

    if _is_error(result):
        return result
    return result[0], 200, {"Content-Type": "application/json"}


def suggest_batch(
    project_id: str,
    body: dict[str, list],
    **query_parameters,
) -> list[dict[str, Any]] | ConnexionResponse:
    """suggest subjects for the given documents and return a list of dicts with results
    formatted according to OpenAPI spec"""

    documents = body["documents"]
    result = _suggest(project_id, documents, query_parameters)

    if _is_error(result):
        return result
    for document_results, document in zip(result, documents):
        document_results["document_id"] = document.get("document_id")
    return result, 200, {"Content-Type": "application/json"}


def _suggest(
    project_id: str,
    documents: list[dict[str, str]],
    parameters: dict[str, Any],
) -> list[dict[str, list]] | ConnexionResponse:
    corpus = _documents_to_corpus(documents, subject_index=None)
    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)

    try:
        lang = parameters.get("language") or project.vocab_lang
    except AnnifException as err:
        return server_error(err)

    if lang not in project.vocab.languages:
        return language_not_supported_error(lang)

    limit = parameters.get("limit", 10)
    threshold = parameters.get("threshold", 0.0)

    try:
        suggestion_results = project.suggest_corpus(corpus).filter(limit, threshold)
    except AnnifException as err:
        return server_error(err)

    return suggestion_results_to_list(suggestion_results, project.subjects, lang)


def _documents_to_corpus(
    documents: list[dict[str, Any]],
    subject_index: SubjectIndex | None,
) -> annif.corpus.document.DocumentList:
    if subject_index is not None:
        corpus = [
            Document(
                text=d["text"],
                subject_set=SubjectSet(
                    [subject_index.by_uri(subj["uri"]) for subj in d["subjects"]]
                ),
                metadata=d.get("metadata", {}),
            )
            for d in documents
            if "text" in d and "subjects" in d
        ]
    else:
        corpus = [
            Document(text=d["text"], subject_set=None, metadata=d.get("metadata", {}))
            for d in documents
            if "text" in d
        ]
    return DocumentList(corpus)


def learn(
    project_id: str,
    body: list[dict[str, Any]],
) -> ConnexionResponse | tuple[None, int]:
    """learn from documents and return an empty 204 response if succesful"""

    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)

    try:
        corpus = _documents_to_corpus(body, project.subjects)
        project.learn(corpus)
    except NotEnabledException:
        return learning_not_enabled_error(project_id)
    except AnnifException as err:
        return server_error(err)

    return None, 204, {"Content-Type": "application/json"}
