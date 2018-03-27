"""Registry of backend types for Annif"""

import configparser
from flask import current_app


_backend_types = {}


def register_backend_type(backend_type):
    _backend_types[backend_type.name] = backend_type


def get_backend_type(backend_type):
    try:
        return _backend_types[backend_type]
    except KeyError:
        raise ValueError("No such backend type {}".format(backend_type))


from . import dummy
register_backend_type(dummy.DummyBackend)

from . import http
register_backend_type(http.HTTPBackend)

from . import tfidf
register_backend_type(tfidf.TFIDFBackend)


def create_backends(app, backend_types):
    return backends


def init_backends(app):
    backends_file = app.config['BACKENDS_FILE']
    config = configparser.ConfigParser()
    with open(backends_file) as bef:
        config.read_file(bef)

    # create backend instances from the configuration file
    backends = {}
    for backend_id in config.sections():
        betype_id = config[backend_id]['type']
        try:
            beclass = get_backend_type(betype_id)
        except KeyError:
            raise ValueError("No such backend type {}".format(backend_type))
        backends[backend_id] = beclass(
            backend_id,
            params=config[backend_id],
            datadir=app.config['DATADIR'])

    app.annif_backends = backends


def get_backends():
    """return all backends defined in the backend configuration file"""
    return current_app.annif_backends


def get_backend(backend_id):
    """return a single backend by ID"""

    try:
        return current_app.annif_backends[backend_id]
    except KeyError:
        raise ValueError("No such backend {}".format(backend_id))
