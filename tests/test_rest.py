"""Unit tests for REST API backend code in Annif"""

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


def test_rest_suggest_public(app):
    # public projects should be accessible via REST
    with app.app_context():
        result = annif.rest.suggest(
            'dummy-fi',
            text='example text',
            limit=10,
            threshold=0.0)
        assert 'results' in result


def test_rest_suggest_hidden(app):
    # hidden projects should be accessible if you know the project id
    with app.app_context():
        result = annif.rest.suggest(
            'dummy-en',
            text='example text',
            limit=10,
            threshold=0.0)
        assert 'results' in result


def test_rest_suggest_private(app):
    # private projects should not be accessible via REST
    with app.app_context():
        result = annif.rest.suggest(
            'dummydummy',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 404


def test_rest_suggest_nonexistent(app):
    with app.app_context():
        result = annif.rest.suggest(
            'nonexistent',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 404


def test_rest_suggest_novocab(app):
    with app.app_context():
        result = annif.rest.suggest(
            'novocab',
            text='example text',
            limit=10,
            threshold=0.0)
        assert result.status_code == 503


def test_rest_learn_empty(app):
    with app.app_context():
        response = annif.rest.learn('dummy-en', [])
        assert response == (None, 204)  # success, no output


def test_rest_learn(app):
    documents = [{'text': 'the quick brown fox',
                  'subjects': [{'uri': 'http://example.org/fox',
                                'label': 'fox'}]}]
    with app.app_context():
        response = annif.rest.learn('dummy-en', documents)
        assert response == (None, 204)  # success, no output

        result = annif.rest.suggest(
            'dummy-en',
            text='example text',
            limit=10,
            threshold=0.0)
        assert 'results' in result
        assert result['results'][0]['uri'] == 'http://example.org/fox'
        assert result['results'][0]['label'] == 'fox'


def test_rest_learn_novocab(app):
    with app.app_context():
        result = annif.rest.learn('novocab', [])
        assert result.status_code == 503


def test_rest_learn_nonexistent(app):
    with app.app_context():
        result = annif.rest.learn('nonexistent', [])
        assert result.status_code == 404


def test_rest_learn_not_supported(app):
    with app.app_context():
        result = annif.rest.learn('tfidf-fi', [])
        assert result.status_code == 503
