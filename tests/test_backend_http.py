"""Unit tests for the HTTP backend in Annif"""

import requests.exceptions
import unittest.mock
import annif.backend.http


def test_http_suggest(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = [
            {'uri': 'http://example.org/http', 'label': 'http', 'score': 1.0}]
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            datadir=app.config['DATADIR'])
        result = http.suggest('this is some text', project=project)
        assert len(result) == 1
        assert result[0].uri == 'http://example.org/http'
        assert result[0].label == 'http'
        assert result[0].score == 1.0


def test_http_suggest_with_results(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {'results': [
            {'uri': 'http://example.org/http', 'label': 'http', 'score': 1.0}]}
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            params={
                'endpoint': 'http://api.example.org/dummy/analyze',
            },
            datadir=app.config['DATADIR'])
        result = http.suggest('this is some text', project=project)
        assert len(result) == 1
        assert result[0].uri == 'http://example.org/http'
        assert result[0].label == 'http'
        assert result[0].score == 1.0


def test_http_suggest_zero_score(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = [
            {'uri': 'http://example.org/http', 'label': 'http', 'score': 0.0}]
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            datadir=app.config['DATADIR'])
        result = http.suggest('this is some text', project=project)
        assert len(result) == 0


def test_http_suggest_error(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        mock_request.side_effect = requests.exceptions.RequestException(
            'failed')

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            datadir=app.config['DATADIR'])
        result = http.suggest('this is some text', project=project)
        assert len(result) == 0


def test_http_suggest_json_fails(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.side_effect = ValueError("JSON decode failed")
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            datadir=app.config['DATADIR'])
        result = http.suggest('this is some text', project=project)
        assert len(result) == 0


def test_http_suggest_unexpected_json(app, project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = ["spanish inquisition"]
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            datadir=app.config['DATADIR'])
        result = http.suggest('this is some text', project=project)
        assert len(result) == 0
