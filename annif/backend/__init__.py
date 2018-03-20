"""Registry of backend types for Annif"""

import configparser
import annif


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


def get_backends():
    """return all backends defined in the backend configuration file"""
    backends_file = annif.cxapp.app.config['BACKENDS_FILE']
    config = configparser.ConfigParser()
    with open(backends_file) as bef:
        config.read_file(bef)

    # create backend instances from the configuration file
    backends = {}
    for backend_id in config.sections():
        betype_id = config[backend_id]['type']
        beclass = get_backend_type(betype_id)
        backends[backend_id] = beclass(backend_id, params=config[backend_id])
    return backends


def get_backend(backend_id):
    """return a single backend by ID"""

    backends = get_backends()
    try:
        return backends[backend_id]
    except KeyError:
        raise ValueError("No such backend {}".format(backend_id))
