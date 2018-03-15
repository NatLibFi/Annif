"""Operations supported by Annif, regardless of whether they are executed
via the CLI or the REST API."""

import sys
import annif
import annif.project


def list_projects():
    """
    List available projects.

    Usage: annif list-projects

    REST equivalent: GET /projects/
    """

    return annif.project.get_projects().values()


def show_project(project_id):
    """
    Show project information.

    Usage: annif show-project <project_id>

    REST equivalent:

    GET /projects/<project_id>
    """

    try:
        return annif.project.get_project(project_id)
    except ValueError:
        return None


def list_subjects(project_id):
    """
    Show all subjects for a project.

    USAGE: annif list-subjects <project_id>

    REST equivalent:

    GET /projects/<project_id>/subjects
    """
    pass


def show_subject(project_id, subject_id):
    """
    Show information about a subject.

    USAGE: annif show-subject <project_id> <subject_id>

    REST equivalent:

    GET /projects/<project_id>/subjects/<subject_id>
    """
    pass


def create_subject(project_id, subject_id):
    """
    Create a new subject, or update an existing one.

    annif create-subject <project_id> <subject_id> <subject.txt

    REST equivalent:

    PUT /projects/<project_id>/subjects/<subject_id>
    """
    pass


def load(project_id, directory, clear):
    """
    Load all subjects from a directory.

    USAGE: annif load <project_id> <directory> [--clear=CLEAR]
    """
    pass


def drop_subject(project_id, subject_id):
    """
    Delete a subject.

    USAGE: annif drop-subject <project_id> <subject_id>

    REST equivalent:

    DELETE /projects/<project_id>/subjects/<subject_id>

    """
    pass


def analyze(project_id, text, limit, threshold):
    """Analyze a document and return a list of AnalysisHit objects."""

    try:
        project = annif.project.get_project(project_id)
    except ValueError:
        return None

    return project.analyze(text, limit, threshold)
