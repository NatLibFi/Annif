"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the Swagger specification."""

import annif.project
from annif.hit import HitFilter


def list_projects():
    return [proj.dump() for proj in annif.project.get_projects().values()]


def show_project(project_id):
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return "Project '{}' not found".format(project_id), 404

    return project.dump()


def list_subjects(project_id):
    return annif.operations.list_subjects(project_id)


def show_subject(project_id, subject_id):
    return annif.operations.show_subjects(project_id, subject_id)


def create_subject(project_id, subject_id):
    return annif.operations.create_subject(project_id, subject_id)


def drop_subject(project_id, subject_id):
    return annif.operations.drop_subject(project_id, subject_id)


def analyze(project_id, text, limit, threshold):
    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return "Project '{}' not found".format(project_id), 404

    hit_filter = HitFilter(limit, threshold)
    hits = hit_filter(project.analyze(text))
    return [hit._asdict() for hit in hits]
