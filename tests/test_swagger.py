"""Unit tests for Annif REST API / Swagger spec"""


def test_swagger_cors(app_client):
    # test that the service supports CORS
    req = app_client.get('http://localhost:8000/v1/projects')
    assert req.headers['access-control-allow-origin'] == '*'


def test_swagger_list_projects(app_client):
    req = app_client.get('http://localhost:8000/v1/projects')
    assert req.status_code == 200
    assert 'projects' in req.get_json()


def test_swagger_show_project(app_client):
    req = app_client.get('http://localhost:8000/v1/projects/dummy-fi')
    assert req.status_code == 200
    assert req.get_json()['project_id'] == 'dummy-fi'


def test_swagger_show_project_nonexistent(app_client):
    req = app_client.get('http://localhost:8000/v1/projects/nonexistent')
    assert req.status_code == 404


def test_swagger_suggest(app_client):
    data = {'text': 'example text'}
    req = app_client.post(
        'http://localhost:8000/v1/projects/dummy-fi/suggest', data=data)
    assert req.status_code == 200
    assert 'results' in req.get_json()


def test_swagger_suggest_nonexistent(app_client):
    data = {'text': 'example text'}
    req = app_client.post(
        'http://localhost:8000/v1/projects/nonexistent/suggest', data=data)
    assert req.status_code == 404


def test_swagger_suggest_novocab(app_client):
    data = {'text': 'example text'}
    req = app_client.post(
        'http://localhost:8000/v1/projects/novocab/suggest', data=data)
    assert req.status_code == 503


def test_swagger_learn(app_client):
    data = [{'text': 'the quick brown fox',
            'subjects': [{'uri': 'http://example.org/fox', 'label': 'fox'}]}]
    req = app_client.post(
        'http://localhost:8000/v1/projects/dummy-fi/learn', json=data)
    assert req.status_code == 204


def test_swagger_learn_nonexistent(app_client):
    data = []
    req = app_client.post(
        'http://localhost:8000/v1/projects/nonexistent/learn', json=data)
    assert req.status_code == 404


def test_swagger_learn_novocab(app_client):
    data = []
    req = app_client.post(
        'http://localhost:8000/v1/projects/novocab/learn', json=data)
    assert req.status_code == 503
