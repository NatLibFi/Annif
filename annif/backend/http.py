"""HTTP/REST client backend that makes calls to a web service
and returns the results"""


import requests
from annif.hit import AnalysisHit, ListAnalysisResult
from . import backend


class HTTPBackend(backend.AnnifBackend):
    name = "http"

    def _analyze(self, text, project, params):
        data = {'text': text}
        if 'project' in params:
            data['project'] = params['project']
        req = requests.post(params['endpoint'], data=data)
        response = req.json()
        if 'results' in response:
            results = response['results']
        else:
            results = response
        return ListAnalysisResult([AnalysisHit(uri=h['uri'],
                                               label=h['label'],
                                               score=h['score'])
                                   for h in results
                                   if h['score'] > 0.0],
                                  project.subjects)
