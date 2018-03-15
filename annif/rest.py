"""Definitions for REST API operations. These are wired via Connexion to
methods defined in the Swagger specification."""

import annif.operations


def list_projects():
    return [proj.dump() for proj in annif.operations.list_projects()]


def show_project(project_id):
    project = annif.operations.show_project(project_id)
    if project is not None:
        return project.dump()

    return "Project '{}' not found".format(project_id), 404


def list_subjects(project_id):
    return annif.operations.list_subjects(project_id)


def show_subject(project_id, subject_id):
    return annif.operations.show_subjects(project_id, subject_id)


def create_subject(project_id, subject_id):
    return annif.operations.create_subject(project_id, subject_id)


def drop_subject(project_id, subject_id):
    return annif.operations.drop_subject(project_id, subject_id)


def analyze(project_id, text, limit, threshold):
    hits = annif.operations.analyze(project_id, text, limit, threshold)
    if hits is None:
        return "Project '{}' not found".format(project_id), 404

    return [hit.dump() for hit in hits]
