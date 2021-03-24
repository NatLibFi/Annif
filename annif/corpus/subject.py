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
        yield Subject(uri=clean_uri, label=label, notation=notation, text=None)

    @property
    def subjects(self):
        with open(self.path, encoding='utf-8-sig') as subjfile:
            for line in subjfile:
                yield from self._parse_line(line)

    def save_skos(self, path, language):
        """Save the contents of the subject vocabulary into a SKOS/Turtle
        file with the given path name."""
        serialize_subjects_to_skos(self.subjects, language, path)


class SubjectIndex:
    """An index that remembers the associations between integers subject IDs
    and their URIs and labels."""

    def __init__(self, corpus=None):
        """Initialize the subject index from a subject corpus."""
        self._uris = []
        self._labels = []
        self._notations = []
        self._uri_idx = {}
        self._label_idx = {}
        if corpus is not None:
            for subject_id, subject in enumerate(corpus.subjects):
                self._append(subject_id, subject.uri, subject.label,
                             subject.notation)

    def __len__(self):
        return len(self._uris)

    def __getitem__(self, subject_id):
        return (self._uris[subject_id], self._labels[subject_id],
                self._notations[subject_id])

    def _append(self, subject_id, uri, label, notation):
        self._uris.append(uri)
        self._labels.append(label)
        self._notations.append(notation)
        self._uri_idx[uri] = subject_id
        self._label_idx[label] = subject_id

    def append(self, uri, label, notation):
        subject_id = len(self._uris)
        self._append(subject_id, uri, label, notation)

    def contains_uri(self, uri):
        return uri in self._uris

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

    def uris_to_labels(self, uris):
        """return a list of labels corresponding to the given URIs; unknown
        URIs are ignored"""

        return [self[subject_id][1]
                for subject_id in (self.by_uri(uri) for uri in uris)
                if subject_id is not None]

    def labels_to_uris(self, labels):
        """return a list of URIs corresponding to the given labels; unknown
        labels are ignored"""

        return [self[subject_id][0]
                for subject_id in (self.by_label(label) for label in labels)
                if subject_id is not None]

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
        return cls(corpus)


class SubjectSet:
    """Represents a set of subjects for a document."""

    def __init__(self, subj_data=None):
        """Create a SubjectSet and optionally initialize it from a tuple
        (URIs, labels)"""

        uris, labels = subj_data or ([], [])
        self.subject_uris = set(uris)
        self.subject_labels = set(labels)

    @classmethod
    def from_string(cls, subj_data):
        sset = cls()
        for line in subj_data.splitlines():
            sset._parse_line(line)
        return sset

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
        return len(self.subject_uris) >= len(self.subject_labels)

    def as_vector(self, subject_index, destination=None, warnings=True):
        """Return the hits as a one-dimensional NumPy array in sklearn
           multilabel indicator format, using a subject index as the source
           of subjects. Use destination array if given (not None), otherwise
           create and return a new one. If warnings=True, log warnings for
           unknown URIs."""

        if destination is None:
            destination = np.zeros(len(subject_index), dtype=bool)

        if self.has_uris():
            for uri in self.subject_uris:
                subject_id = subject_index.by_uri(
                    uri, warnings=warnings)
                if subject_id is not None:
                    destination[subject_id] = True
        else:
            for label in self.subject_labels:
                subject_id = subject_index.by_label(label)
                if subject_id is not None:
                    destination[subject_id] = True
        return destination
