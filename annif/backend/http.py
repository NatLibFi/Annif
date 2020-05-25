"""HTTP/REST client backend that makes calls to a web service
and returns the results"""


import requests
import requests.exceptions
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from . import backend


class HTTPBackend(backend.AnnifBackend):
    name = "http"

    @property
    def is_trained(self):
        response = self._get_project_info()
        if 'is_trained' in response:
            return response['is_trained']
        else:
            return None

    @property
    def modification_time(self):
        response = self._get_project_info()
        if 'modification_time' in response:
            return response['modification_time']
        else:
            return None

    def _get_project_info(self):
        params = self._get_backend_params(None)
        try:
            req = requests.get(params['endpoint'].replace('/suggest', ''))
            req.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.warning("HTTP request failed: {}".format(err))
            return {}
        try:
            return req.json()
        except ValueError as err:
            self.warning("JSON decode failed: {}".format(err))
            return {}

    def _suggest(self, text, params):
        data = {'text': text}
        if 'project' in params:
            data['project'] = params['project']

        try:
            req = requests.post(params['endpoint'], data=data)
            req.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.warning("HTTP request failed: {}".format(err))
            return ListSuggestionResult([], self.project.subjects)

        try:
            response = req.json()
        except ValueError as err:
            self.warning("JSON decode failed: {}".format(err))
            return ListSuggestionResult([], self.project.subjects)

        if 'results' in response:
            results = response['results']
        else:
            results = response

        try:
            subject_suggestions = [SubjectSuggestion(
                uri=hit['uri'],
                label=None,
                notation=None,
                score=hit['score'])
                for hit in results if hit['score'] > 0.0]
        except (TypeError, ValueError) as err:
            self.warning("Problem interpreting JSON data: {}".format(err))
            return ListSuggestionResult([], self.project.subjects)

        return ListSuggestionResult.create_from_index(subject_suggestions,
                                                      self.project.subjects)
