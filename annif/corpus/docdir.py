"""A directory of files as a full text document corpus"""


import glob
import os.path
import re


class DocumentDirectory:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        """Iterate through the directory, yielding tuples of (docfile, keyfile) containing file paths.
        If there is no key file, the keyfile will be None."""

        for filename in glob.glob(os.path.join(self.path, '*.txt')):
            keyfilename = re.sub(r'\.txt$', '.key', filename)
            if not os.path.exists(keyfilename):
                keyfilename = None
            yield (filename, keyfilename)
