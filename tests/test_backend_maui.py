"""Unit tests for the HTTP backend in Annif"""

import requests.exceptions
import responses
import unittest.mock
import annif.backend.maui


@responses.activate
def test_maui_suggest(app, project):
    responses.add(responses.POST,
                  'http://api.example.org/mauiservice/dummy/suggest',
                  json={'title': '1 recommendation from dummy',
                        'topics': [{'id': 'http://example.org/maui',
                                    'label': 'maui',
                                    'probability': 1.0}]})

    maui_type = annif.backend.get_backend("maui")
    maui = maui_type(
        backend_id='maui',
        config_params={
            'endpoint': 'http://api.example.org/mauiservice/',
            'tagger': 'dummy'},
        datadir=app.config['DATADIR'])
    result = maui.suggest('this is some text', project=project)
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/maui'
    assert result[0].label == 'maui'
    assert result[0].score == 1.0
    assert len(responses.calls) == 1


@responses.activate
def test_maui_suggest_zero_score(app, project):
    responses.add(responses.POST,
                  'http://api.example.org/mauiservice/dummy/suggest',
                  json={'title': '1 recommendation from dummy',
                        'topics': [{'id': 'http://example.org/maui',
                                    'label': 'maui',
                                    'probability': 0.0}]})
    maui_type = annif.backend.get_backend("maui")
    maui = maui_type(
        backend_id='maui',
        config_params={
            'endpoint': 'http://api.example.org/mauiservice/',
            'tagger': 'dummy'},
        datadir=app.config['DATADIR'])
    result = maui.suggest('this is some text', project=project)
    assert len(result) == 0
    assert len(responses.calls) == 1


def test_maui_suggest_error(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        mock_request.side_effect = requests.exceptions.RequestException(
            'failed')

        maui_type = annif.backend.get_backend("maui")
        maui = maui_type(
            backend_id='maui',
            config_params={
                'endpoint': 'http://api.example.org/mauiservice/',
                'tagger': 'dummy'},
            datadir=app.config['DATADIR'])
        result = maui.suggest('this is some text', project=project)
        assert len(result) == 0


def test_maui_suggest_json_fails(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.side_effect = ValueError("JSON decode failed")
        mock_request.return_value = mock_response

        maui_type = annif.backend.get_backend("maui")
        maui = maui_type(
            backend_id='maui',
            config_params={
                'endpoint': 'http://api.example.org/mauiservice/',
                'tagger': 'dummy'},
            datadir=app.config['DATADIR'])
        result = maui.suggest('this is some text', project=project)
        assert len(result) == 0


@responses.activate
def test_maui_suggest_unexpected_json(app, project):
    responses.add(responses.POST,
                  'http://api.example.org/mauiservice/dummy/suggest',
                  json=["spanish inquisition"])

    maui_type = annif.backend.get_backend("maui")
    maui = maui_type(
        backend_id='maui',
        config_params={
            'endpoint': 'http://api.example.org/mauiservice/',
            'tagger': 'dummy'},
        datadir=app.config['DATADIR'])
    result = maui.suggest('this is some text', project=project)
    assert len(result) == 0
    assert len(responses.calls) == 1
