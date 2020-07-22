"""HTTP/REST client backend that makes calls to a web service
and returns the results"""


import dateutil.parser
import requests
import requests.exceptions
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from annif.exception import OperationFailedException
from . import backend


class HTTPBackend(backend.AnnifBackend):
    name = "http"

    @property
    def is_trained(self):
        return self._get_project_info('is_trained')

    @property
    def modification_time(self):
        mtime = self._get_project_info('modification_time')
        if mtime is None:
            return None
        return dateutil.parser.parse(mtime)

    def _get_project_info(self, key):
        params = self._get_backend_params(None)
        try:
            req = requests.get(params['endpoint'].replace('/suggest', ''))
            req.raise_for_status()
        except requests.exceptions.RequestException as err:
            msg = f"HTTP request failed: {err}"
            raise OperationFailedException(msg) from err
        try:
            response = req.json()
        except ValueError as err:
            msg = f"JSON decode failed: {err}"
            raise OperationFailedException(msg) from err

        if key in response:
            return response[key]
        else:
            return None

    def _suggest(self, text, params):
        data = {'text': text}
        if 'project' in params:
            data['project'] = params['project']

        try:
            req = requests.post(params['endpoint'], data=data)
            req.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.warning("HTTP request failed: {}".format(err))
            return ListSuggestionResult([])

        try:
            response = req.json()
        except ValueError as err:
            self.warning("JSON decode failed: {}".format(err))
            return ListSuggestionResult([])

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
            return ListSuggestionResult([])

        return ListSuggestionResult.create_from_index(subject_suggestions,
                                                      self.project.subjects)
