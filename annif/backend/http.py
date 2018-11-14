"""HTTP/REST client backend that makes calls to a web service
and returns the results"""


import requests
import requests.exceptions
from annif.hit import AnalysisHit, ListAnalysisResult
from . import backend


class HTTPBackend(backend.AnnifBackend):
    name = "http"

    def _analyze(self, text, project, params):
        data = {'text': text}
        if 'project' in params:
            data['project'] = params['project']

        try:
            req = requests.post(params['endpoint'], data=data)
            req.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.warning("HTTP request failed: {}".format(err))
            return ListAnalysisResult([], project.subjects)

        try:
            response = req.json()
        except ValueError as err:
            self.warning("JSON decode failed: {}".format(err))
            return ListAnalysisResult([], project.subjects)

        if 'results' in response:
            results = response['results']
        else:
            results = response

        try:
            return ListAnalysisResult([AnalysisHit(uri=h['uri'],
                                                   label=h['label'],
                                                   score=h['score'])
                                       for h in results
                                       if h['score'] > 0.0],
                                      project.subjects)
        except (TypeError, ValueError) as err:
            self.warning("Problem interpreting JSON data: {}".format(err))
            return ListAnalysisResult([], project.subjects)
