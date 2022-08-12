"""Classes for supporting subject corpora expressed as directories or files"""

import annif.util
import numpy as np
from annif import logger
from .types import Subject
from .skos import serialize_subjects_to_skos


class SubjectFileTSV:
    """A subject vocabulary stored in a TSV file."""

    def __init__(self, path):
        self.path = path

    def _parse_line(self, line):
        vals = line.strip().split('\t', 2)
        clean_uri = annif.util.cleanup_uri(vals[0])
        label = vals[1] if len(vals) >= 2 else None
        notation = vals[2] if len(vals) >= 3 else None
        yield Subject(uri=clean_uri, label=label, notation=notation)

    @property
    def languages(self):
        # we don't have information about the language(s) of labels
        return None

    def subjects(self, language):
        with open(self.path, encoding='utf-8-sig') as subjfile:
            for line in subjfile:
                yield from self._parse_line(line)

    def save_skos(self, path, language):
        """Save the contents of the subject vocabulary into a SKOS/Turtle
        file with the given path name."""
        serialize_subjects_to_skos(self.subjects(language), language, path)


class SubjectIndex:
    """An index that remembers the associations between integers subject IDs
    and their URIs and labels."""

    def __init__(self):
        self._uris = []
        self._labels = []
        self._notations = []
        self._uri_idx = {}
        self._label_idx = {}

    def load_subjects(self, corpus, language):
        """Initialize the subject index from a subject corpus using labels
        in the given language."""

        for subject_id, subject in enumerate(corpus.subjects(language)):
            self._append(subject_id, subject)

    def __len__(self):
        return len(self._uris)

    def __getitem__(self, subject_id):
        return Subject(uri=self._uris[subject_id],
                       label=self._labels[subject_id],
                       notation=self._notations[subject_id])

    def _append(self, subject_id, subject):
        self._uris.append(subject.uri)
        self._labels.append(subject.label)
        self._notations.append(subject.notation)
        self._uri_idx[subject.uri] = subject_id
        self._label_idx[subject.label] = subject_id

    def append(self, subject):
        subject_id = len(self._uris)
        self._append(subject_id, subject)

    def contains_uri(self, uri):
        return uri in self._uri_idx

    def by_uri(self, uri, warnings=True):
        """return the subject index of a subject by its URI, or None if not found.
        If warnings=True, log a warning message if the URI cannot be found."""
        try:
            return self._uri_idx[uri]
        except KeyError:
            if warnings:
                logger.warning('Unknown subject URI <%s>', uri)
            return None

    def by_label(self, label):
        """return the subject index of a subject by its label"""
        try:
            return self._label_idx[label]
        except KeyError:
            logger.warning('Unknown subject label "%s"', label)
            return None

    def deprecated_ids(self):
        """return indices of deprecated subjects"""

        return [subject_id for subject_id, label in enumerate(self._labels)
                if label is None]

    @property
    def active(self):
        """return a list of (subject_id, uri, label, notation) tuples of all
        subjects that are not deprecated"""

        return [(subj_id, uri, label, notation)
                for subj_id, (uri, label, notation)
                in enumerate(zip(self._uris, self._labels, self._notations))
                if label is not None]

    def save(self, path):
        """Save this subject index into a file."""

        with open(path, 'w', encoding='utf-8') as subjfile:
            for uri, label, notation in self:
                line = "<{}>".format(uri)
                if label is not None:
                    line += ('\t' + label)
                    if notation is not None:
                        line += ('\t' + notation)
                print(line, file=subjfile)

    @classmethod
    def load(cls, path):
        """Load a subject index from a TSV file and return it."""

        corpus = SubjectFileTSV(path)
        subject_index = cls()
        subject_index.load_subjects(corpus, None)
        return subject_index


class SubjectSet:
    """Represents a set of subjects for a document."""

    def __init__(self, subject_ids=None):
        """Create a SubjectSet and optionally initialize it from an iterable
        of subject IDs"""

        if subject_ids:
            # use set comprehension to eliminate possible duplicates
            self._subject_ids = list({subject_id
                                      for subject_id in subject_ids
                                      if subject_id is not None})
        else:
            self._subject_ids = []

    def __len__(self):
        return len(self._subject_ids)

    def __getitem__(self, idx):
        return self._subject_ids[idx]

    def __bool__(self):
        return bool(self._subject_ids)

    def __eq__(self, other):
        if isinstance(other, SubjectSet):
            return self._subject_ids == other._subject_ids

        return False

    @classmethod
    def from_string(cls, subj_data, subject_index):
        subject_ids = set()
        for line in subj_data.splitlines():
            uri, label = cls._parse_line(line)
            if uri is not None:
                subject_ids.add(subject_index.by_uri(uri))
            else:
                subject_ids.add(subject_index.by_label(label))
        return cls(subject_ids)

    @staticmethod
    def _parse_line(line):
        uri = label = None
        vals = line.split("\t")
        for val in vals:
            val = val.strip()
            if val == '':
                continue
            if val.startswith('<') and val.endswith('>'):  # URI
                uri = val[1:-1]
                continue
            label = val
            break
        return uri, label

    def as_vector(self, size=None, destination=None):
        """Return the hits as a one-dimensional NumPy array in sklearn
           multilabel indicator format. Use destination array if given (not
           None), otherwise create and return a new one of the given size."""

        if destination is None:
            destination = np.zeros(size, dtype=bool)

        destination[list(self._subject_ids)] = True

        return destination
