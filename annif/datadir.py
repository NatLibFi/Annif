"""Mixin class for types that need a data directory"""
from __future__ import annotations

import os
import os.path


class DatadirMixin:
    """Mixin class for types that need a data directory for storing files"""

    def __init__(self, datadir: str, typename: str, identifier: str) -> None:
        self._datadir_path = os.path.join(datadir, typename, identifier)

    @property
    def datadir(self) -> str:
        if not os.path.exists(self._datadir_path):
            try:
                os.makedirs(self._datadir_path)
            except FileExistsError:
                # apparently the datadir was created by another thread!
                pass
        return self._datadir_path
