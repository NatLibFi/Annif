"""Registry of backend types for Annif"""

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
