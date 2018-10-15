"""Unit tests for REST API backend code in Annif"""

import connexion
import annif.rest


def test_rest_novocab(app):
    with app.app_context():
        result = annif.rest.analyze(
            'novocab',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 503
