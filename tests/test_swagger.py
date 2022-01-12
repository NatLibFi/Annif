"""Unit tests for Annif REST API / Swagger spec"""

import requests


def test_swagger_cors(app_with_server):
    # fixture needed once to start server in background
    # test that the service supports CORS
    req = requests.get('http://localhost:8000/v1/projects')
    assert req.headers['access-control-allow-origin'] == '*'


def test_swagger_list_projects():
    req = requests.get('http://localhost:8000/v1/projects')
    assert req.status_code == 200
    assert 'projects' in req.json()


def test_swagger_show_project():
    req = requests.get('http://localhost:8000/v1/projects/dummy-fi')
    assert req.status_code == 200
    assert req.json()['project_id'] == 'dummy-fi'


def test_swagger_show_project_nonexistent():
    req = requests.get('http://localhost:8000/v1/projects/nonexistent')
    assert req.status_code == 404


def test_swagger_suggest():
    data = {'text': 'example text'}
    req = requests.post('http://localhost:8000/v1/projects/dummy-fi/suggest',
                        data=data)
    assert req.status_code == 200
    assert 'results' in req.json()


def test_swagger_suggest_nonexistent():
    data = {'text': 'example text'}
    req = requests.post(
        'http://localhost:8000/v1/projects/nonexistent/suggest', data=data)
    assert req.status_code == 404


def test_swagger_suggest_novocab():
    data = {'text': 'example text'}
    req = requests.post('http://localhost:8000/v1/projects/novocab/suggest',
                        data=data)
    assert req.status_code == 503


def test_swagger_learn():
    data = [{'text': 'the quick brown fox',
            'subjects': [{'uri': 'http://example.org/fox', 'label': 'fox'}]}]
    req = requests.post('http://localhost:8000/v1/projects/dummy-fi/learn',
                        json=data)
    assert req.status_code == 204


def test_swagger_learn_nonexistent():
    data = []
    req = requests.post('http://localhost:8000/v1/projects/nonexistent/learn',
                        json=data)
    assert req.status_code == 404


def test_swagger_learn_novocab():
    data = []
    req = requests.post('http://localhost:8000/v1/projects/novocab/learn',
                        json=data)
    assert req.status_code == 503
