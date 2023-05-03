"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the OpenAPI specification."""

import importlib

import connexion

import annif.registry
from annif.corpus import Document, DocumentList, SubjectSet
from annif.exception import AnnifException
from annif.project import Access


def project_not_found_error(project_id):
    """return a Connexion error object when a project is not found"""

    return connexion.problem(
        status=404,
        title="Project not found",
        detail="Project '{}' not found".format(project_id),
    )


def server_error(err):
    """return a Connexion error object when there is a server error (project
    or backend problem)"""

    return connexion.problem(
        status=503, title="Service unavailable", detail=err.format_message()
    )


def show_info():
    """return version of annif and a title for the api according to OpenAPI spec"""

    result = {"title": "Annif REST API", "version": importlib.metadata.version("annif")}
    return result, 200, {"Content-Type": "application/json"}


def language_not_supported_error(lang):
    """return a Connexion error object when attempting to use unsupported language"""

    return connexion.problem(
        status=400,
        title="Bad Request",
        detail=f'language "{lang}" not supported by vocabulary',
    )


def list_projects():
    """return a dict with projects formatted according to OpenAPI spec"""

    result = {
        "projects": [
            proj.dump()
            for proj in annif.registry.get_projects(min_access=Access.public).values()
        ]
    }
    return result, 200, {"Content-Type": "application/json"}


def show_project(project_id):
    """return a single project formatted according to OpenAPI spec"""

    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)
    return project.dump(), 200, {"Content-Type": "application/json"}


def _suggestion_to_dict(suggestion, subject_index, language):
    subject = subject_index[suggestion.subject_id]
    return {
        "uri": subject.uri,
        "label": subject.labels[language],
        "notation": subject.notation,
        "score": suggestion.score,
    }


def _hit_sets_to_list(hit_sets, subjects, lang):
    return [
        {"results": [_suggestion_to_dict(hit, subjects, lang) for hit in hits]}
        for hits in hit_sets
    ]


def _is_error(result):
    return (
        isinstance(result, connexion.lifecycle.ConnexionResponse)
        and result.status_code >= 400
    )


def suggest(project_id, body):
    """suggest subjects for the given text and return a dict with results
    formatted according to OpenAPI spec"""

    parameters = dict(
        (key, body[key]) for key in ["language", "limit", "threshold"] if key in body
    )
    documents = [{"text": body["text"]}]
    result = _suggest(project_id, documents, parameters)

    if _is_error(result):
        return result
    return result[0], 200, {"Content-Type": "application/json"}


def suggest_batch(project_id, body, **query_parameters):
    """suggest subjects for the given documents and return a list of dicts with results
    formatted according to OpenAPI spec"""

    documents = body["documents"]
    result = _suggest(project_id, documents, query_parameters)

    if _is_error(result):
        return result
    for document_results, document in zip(result, documents):
        document_results["document_id"] = document.get("document_id")
    return result, 200, {"Content-Type": "application/json"}


def _suggest(project_id, documents, parameters):
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
        hit_sets = project.suggest_corpus(corpus).filter(limit, threshold)
    except AnnifException as err:
        return server_error(err)

    return _hit_sets_to_list(hit_sets, project.subjects, lang)


def _documents_to_corpus(documents, subject_index):
    if subject_index is not None:
        corpus = [
            Document(
                text=d["text"],
                subject_set=SubjectSet(
                    [subject_index.by_uri(subj["uri"]) for subj in d["subjects"]]
                ),
            )
            for d in documents
            if "text" in d and "subjects" in d
        ]
    else:
        corpus = [
            Document(text=d["text"], subject_set=None) for d in documents if "text" in d
        ]
    return DocumentList(corpus)


def learn(project_id, body):
    """learn from documents and return an empty 204 response if succesful"""

    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)

    try:
        corpus = _documents_to_corpus(body, project.subjects)
        project.learn(corpus)
    except AnnifException as err:
        return server_error(err)

    return None, 204, {"Content-Type": "application/json"}
