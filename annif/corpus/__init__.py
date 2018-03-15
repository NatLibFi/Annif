"""Annif corpus operations"""


from .docdir import DocumentDirectory


class SubjectSet:
    def __init__(self, subj_data):
        self.subject_uris = set()
        self.subject_labels = set()
        self._parse(subj_data)

    def _parse(self, subj_data):
        for line in subj_data.splitlines():
            self._parse_line(line)

    def _parse_line(self, line):
        vals = line.split("\t")
        for val in vals:
            val = val.strip()
            if val == '':
                continue
            if val.startswith('<') and val.endswith('>'):  # URI
                self.subject_uris.add(val[1:-1])
                continue
            self.subject_labels.add(val)
            return

    def has_uris(self):
        """returns True if the URIs for all subjects are known"""
        return len(self.subject_uris) == len(self.subject_labels)
