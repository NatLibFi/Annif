"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the Swagger specification."""

import connexion
import annif.project
from annif.hit import HitFilter


def project_not_found_error(project_id):
    return connexion.problem(
        status=404,
        title='Project not found',
        detail="Project '{}' not found".format(project_id))


def list_projects():
    return {'projects': [proj.dump()
                         for proj in annif.project.get_projects().values()]}


def show_project(project_id):
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return project_not_found_error(project_id)
    return project.dump()


def analyze(project_id, text, limit, threshold):
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return project_not_found_error(project_id)

    hit_filter = HitFilter(limit, threshold)
    hits = hit_filter(project.analyze(text))
    return {'results': [hit._asdict() for hit in hits]}
