"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the Swagger specification."""

import connexion
import annif.project
from annif.hit import HitFilter
from annif.exception import AnnifException


def project_not_found_error(project_id):
    """return a Connexion error object when a project is not found"""

    return connexion.problem(
        status=404,
        title='Project not found',
        detail="Project '{}' not found".format(project_id))


def server_error(err):
    """return a Connexion error object when there is a server error (project
    or backend problem)"""

    return connexion.problem(
        status=503,
        title='Service unavailable',
        detail=err.format_message())


def list_projects():
    """return a dict with projects formatted according to Swagger spec"""

    return {'projects': [proj.dump()
                         for proj in annif.project.get_projects().values()]}


def show_project(project_id):
    """return a single project formatted according to Swagger spec"""

    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return project_not_found_error(project_id)
    return project.dump()


def analyze(project_id, text, limit, threshold):
    """analyze text and return a dict with results formatted according to
    Swagger spec"""

    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return project_not_found_error(project_id)

    hit_filter = HitFilter(limit, threshold)
    try:
        result = project.analyze(text)
    except AnnifException as err:
        return server_error(err)
    hits = hit_filter(result)
    return {'results': [hit._asdict() for hit in hits]}
