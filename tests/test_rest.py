"""Unit tests for REST API backend code in Annif"""

import connexion
import annif.rest


def test_rest_list_projects(app):
    with app.app_context():
        result = annif.rest.list_projects()
        project_ids = [proj['project_id'] for proj in result['projects']]
        # public project should be returned
        assert 'dummy-fi' in project_ids
        # hidden project should not be returned
        assert 'dummy-en' not in project_ids
        # private project should not be returned
        assert 'dummydummy' not in project_ids
        # project with no access level setting should be returned
        assert 'ensemble' in project_ids


def test_rest_show_project_public(app):
    # public projects should be accessible via REST
    with app.app_context():
        result = annif.rest.show_project('dummy-fi')
        assert result['project_id'] == 'dummy-fi'


def test_rest_show_project_hidden(app):
    # hidden projects should be accessible if you know the project id
    with app.app_context():
        result = annif.rest.show_project('dummy-en')
        assert result['project_id'] == 'dummy-en'


def test_rest_show_project_private(app):
    # private projects should not be accessible via REST
    with app.app_context():
        result = annif.rest.show_project('dummydummy')
        assert result.status_code == 404


def test_rest_show_project_nonexistent(app):
    with app.app_context():
        result = annif.rest.show_project('nonexistent')
        assert result.status_code == 404


def test_rest_analyze_public(app):
    # public projects should be accessible via REST
    with app.app_context():
        result = annif.rest.analyze(
            'dummy-fi',
            text='example text',
            limit=10,
            threshold=0.0)
        assert 'results' in result


def test_rest_analyze_hidden(app):
    # hidden projects should be accessible if you know the project id
    with app.app_context():
        result = annif.rest.analyze(
            'dummy-en',
            text='example text',
            limit=10,
            threshold=0.0)
        assert 'results' in result


def test_rest_analyze_private(app):
    # private projects should not be accessible via REST
    with app.app_context():
        result = annif.rest.analyze(
            'dummydummy',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 404


def test_rest_analyze_nonexistent(app):
    with app.app_context():
        result = annif.rest.analyze(
            'nonexistent',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 404


def test_rest_novocab(app):
    with app.app_context():
        result = annif.rest.analyze(
            'novocab',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 503
