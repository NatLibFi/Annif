"""HTTP/REST client backend that makes calls to a web service
and returns the results"""


import requests
import requests.exceptions
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from . import backend


class HTTPBackend(backend.AnnifBackend):
    name = "http"

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
            hits = [(hit['uri'], float(hit['score'])) for hit in results]
        except (TypeError, ValueError) as err:
            self.warning("Problem interpreting JSON data: {}".format(err))
            return ListSuggestionResult([], self.project.subjects)

        subject_suggestions = []
        for uri, score in hits:
            if score > 0.0:
                subject = self.project.subjects[
                    self.project.subjects.by_uri(uri)]
                subject_suggestions.append(
                    SubjectSuggestion(uri=uri,
                                      label=subject[1],
                                      notation=subject[2],
                                      score=score))
        return ListSuggestionResult(subject_suggestions, self.project.subjects)
