"""Registry of backend types for Annif"""

import configparser
from flask import current_app
from . import dummy
from . import ensemble
from . import http
from . import tfidf
from . import fasttext
from . import pav


_backend_types = {}


def register_backend(backend):
    _backend_types[backend.name] = backend


def get_backend(backend_id):
    try:
        return _backend_types[backend_id]
    except KeyError:
        raise ValueError("No such backend type {}".format(backend_id))


register_backend(dummy.DummyBackend)
register_backend(ensemble.EnsembleBackend)
register_backend(http.HTTPBackend)
register_backend(tfidf.TFIDFBackend)
register_backend(fasttext.FastTextBackend)
register_backend(pav.PAVBackend)
