"""Annif corpus operations"""


class SubjectSet:
    def __init__(self, subj_data):
        self.subject_uris = set()
        self.subject_labels = set()

        for line in subj_data.splitlines():
            vals = line.split("\t")
            for val in vals:
                val = val.strip()
                if val == '':
                    continue
                if val.startswith('<') and val.endswith('>'):  # URI
                    self.subject_uris.add(val[1:-1])
                    continue
                self.subject_labels.add(val)
                break

    def has_uris(self):
        """returns True if the URIs for all subjects are known"""
        return len(self.subject_uris) == len(self.subject_labels)


from .docdir import DocumentDirectory
