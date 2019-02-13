"""Mixin class for types that need a data directory"""

import os
import os.path


class DatadirMixin:
    """Mixin class for types that need a data directory for storing files"""

    def __init__(self, datadir, typename, identifier):
        self._datadir_path = os.path.join(datadir, typename, identifier)

    @property
    def datadir(self):
        if not os.path.exists(self._datadir_path):
            os.makedirs(self._datadir_path)
        return self._datadir_path
