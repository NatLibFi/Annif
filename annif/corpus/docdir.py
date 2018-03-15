"""A directory of files as a full text document corpus"""


import glob
import os.path
import re


class DocumentDirectory:
    def __init__(self, path, require_keyfile=False):
        self.path = path
        self.require_keyfile = require_keyfile

    def __iter__(self):
        """Iterate through the directory, yielding tuples of (docfile,
        keyfile) containing file paths. If there is no key file and
        require_keyfile is False, the keyfile will be returned as None."""

        for filename in glob.glob(os.path.join(self.path, '*.txt')):
            keyfilename = re.sub(r'\.txt$', '.key', filename)
            if not os.path.exists(keyfilename):
                keyfilename = None
            if keyfilename is None and self.require_keyfile:
                continue
            yield (filename, keyfilename)
