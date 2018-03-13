"""Class representing a single hit from analysis."""


class AnalysisHit:
    """A single hit resulting from analysis."""

    def __init__(self, uri, label, score):
        self.uri = uri
        self.label = label
        self.score = score

    def dump(self):
        """return this object as a dict"""
        return {'uri': self.uri, 'label': self.label, 'score': self.score}
