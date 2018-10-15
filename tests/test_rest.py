"""Unit tests for REST API backend code in Annif"""

import connexion
import annif.rest


def test_rest_show_project_nonexistent(app):
    with app.app_context():
        result = annif.rest.show_project('nonexistent')
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
