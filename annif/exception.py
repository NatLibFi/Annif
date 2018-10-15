"""Custom exceptions used by Annif"""


class AnnifException(Exception):
    """Base Annif exception"""

    def __init__(self, message, project_id=None, backend_id=None):
        self.message = message
        self.project_id = project_id
        self.backend_id = backend_id

    def format_message(self):
        if self.project_id is not None:
            return "Project '{}': {}".format(self.project_id,
                                             self.message)
        if self.backend_id is not None:
            return "Backend '{}': {}".format(self.backend_id,
                                             self.message)
        return "Error: {}".format(self.message)


class NotInitializedException(AnnifException):
    """Exception raised for attempting to use a project or backend that
    cannot be initialized, most likely since it is misconfigured or not yet
    functional because of lack of vocabulary or training."""

    def format_message(self):
        if self.project_id is not None:
            return "Couldn't initialize project '{}': {}".format(self.project_id,
                                                             self.message)
        if self.backend_id is not None:
            return "Couldn't initialize backend '{}': {}".format(self.backend_id,
                                                             self.message)
        return "Couldn't initialize: {}".format(self.message)
