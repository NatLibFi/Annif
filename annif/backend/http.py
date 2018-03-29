"""HTTP/REST client backend that makes calls to a web service
and returns the results"""


import requests
from annif.hit import AnalysisHit
from . import backend


class HTTPBackend(backend.AnnifBackend):
    name = "http"

    def _analyze(self, text, project, params):
        data = {'text': text, 'project': params['project']}
        req = requests.post(params['endpoint'], data=data)
        return [AnalysisHit(h['uri'], h['label'], h['score'])
                for h in req.json()
                if h['score'] > 0.0]
