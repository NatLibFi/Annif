"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the Swagger specification."""

import importlib

import connexion

import annif.registry
from annif.corpus import Document, DocumentList, SubjectSet
from annif.exception import AnnifException
from annif.project import Access
from annif.suggestion import SuggestionFilter


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
    """return version of annif and a title for the api according to Swagger spec"""

    return {"title": "Annif REST API", "version": importlib.metadata.version("annif")}


def list_projects():
    """return a dict with projects formatted according to Swagger spec"""

    return {
        "projects": [
            proj.dump()
            for proj in annif.registry.get_projects(min_access=Access.public).values()
        ]
    }


def show_project(project_id):
    """return a single project formatted according to Swagger spec"""

    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)
    return project.dump()


def _suggestion_to_dict(suggestion, subject_index, language):
    subject = subject_index[suggestion.subject_id]
    return {
        "uri": subject.uri,
        "label": subject.labels[language],
        "notation": subject.notation,
        "score": suggestion.score,
    }


def suggest(project_id, body):
    """suggest subjects for the given text and return a dict with results
    formatted according to Swagger spec"""

    try:
        project = annif.registry.get_project(project_id, min_access=Access.hidden)
    except ValueError:
        return project_not_found_error(project_id)

    try:
        lang = body.get("language") or project.vocab_lang
    except AnnifException as err:
        return server_error(err)

    if lang not in project.vocab.languages:
        return connexion.problem(
            status=400,
            title="Bad Request",
            detail=f'language "{lang}" not supported by vocabulary',
        )

    limit = body.get("limit", 10)
    threshold = body.get("threshold", 0.0)

    try:
        hit_filter = SuggestionFilter(project.subjects, limit, threshold)
        result = project.suggest(body["text"])
    except AnnifException as err:
        return server_error(err)

    hits = hit_filter(result).as_list()
    return {
        "results": [_suggestion_to_dict(hit, project.subjects, lang) for hit in hits]
    }


def _documents_to_corpus(documents, subject_index):
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

    return None, 204
