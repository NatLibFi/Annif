"""Unit tests for the HTTP backend in Annif"""

import pytest
import requests.exceptions
import unittest.mock
import annif.backend.http
from annif.exception import OperationFailedException


def test_http_suggest(app_project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = [
            {'uri': 'http://example.org/dummy', 'label': 'dummy',
             'score': 1.0}]
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=app_project)
        result = http.suggest('this is some text')
        assert len(result) == 1
        hits = result.as_list(app_project.subjects)
        assert hits[0].uri == 'http://example.org/dummy'
        assert hits[0].label == 'dummy'
        assert hits[0].score == 1.0
        assert hits[0].notation is None


def test_http_suggest_with_results(app_project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {'results': [
            {'uri': 'http://example.org/dummy-with-notation', 'label': 'dummy',
             'notation': '42.42', 'score': 1.0}]}
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/dummy/analyze',
            },
            project=app_project)
        http.project.subjects.append(
            'http://example.org/dummy-with-notation', 'dummy', '42.42')

        result = http.suggest('this is some text')
        assert len(result) == 1
        hits = result.as_list(app_project.subjects)
        assert hits[0].uri == 'http://example.org/dummy-with-notation'
        assert hits[0].label == 'dummy'
        assert hits[0].score == 1.0
        assert hits[0].notation == '42.42'


def test_http_suggest_zero_score(project):
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
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        result = http.suggest('this is some text')
        assert len(result) == 0


def test_http_suggest_error(project):
    with unittest.mock.patch('requests.post') as mock_request:
        mock_request.side_effect = requests.exceptions.RequestException(
            'failed')

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        result = http.suggest('this is some text')
        assert len(result) == 0


def test_http_suggest_json_fails(project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.side_effect = ValueError("JSON decode failed")
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        result = http.suggest('this is some text')
        assert len(result) == 0


def test_http_suggest_unexpected_json(project):
    with unittest.mock.patch('requests.post') as mock_request:
        # create a mock response whose .json() method returns the list that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = ["spanish inquisition"]
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        result = http.suggest('this is some text')
        assert len(result) == 0


def test_http_is_trained(project):
    with unittest.mock.patch('requests.get') as mock_request:
        # create a mock response whose .json() method returns the dict that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {'is_trained': True}
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        assert http.is_trained


def test_http_modification_time(project):
    with unittest.mock.patch('requests.get') as mock_request:
        # create a mock response whose .json() method returns the dict that we
        # define here
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {'modification_time':
                                           '1970-1-1 00:00:00'}
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        assert http.modification_time == '1970-1-1 00:00:00'


def test_http_get_project_info_http_error(project):
    with unittest.mock.patch('requests.get') as mock_request:
        mock_request.side_effect = requests.exceptions.RequestException(
            'failed')

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        with pytest.raises(OperationFailedException):
            http._get_project_info('is_trained')


def test_http_get_project_info_json_decode_error(project):
    with unittest.mock.patch('requests.get') as mock_request:
        mock_response = unittest.mock.Mock()
        mock_response.json.side_effect = ValueError("JSON decode failed")
        mock_request.return_value = mock_response

        http_type = annif.backend.get_backend("http")
        http = http_type(
            backend_id='http',
            config_params={
                'endpoint': 'http://api.example.org/analyze',
                'project': 'dummy'},
            project=project)
        with pytest.raises(OperationFailedException):
            http._get_project_info('is_trained')
