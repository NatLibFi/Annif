"""Custom exceptions used by Annif"""


from click import ClickException


class AnnifException(ClickException):
    """Base Annif exception. We define this as a subclass of ClickException so
    that the CLI can automatically handle exceptions."""

    def __init__(self, message, project_id=None, backend_id=None):
        super().__init__(message)
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
    cannot be initialized, most likely since it is not yet functional
    because of lack of vocabulary or training."""

    prefix = "Couldn't initialize"

    def format_message(self):
        if self.project_id is not None:
            return "{} project '{}': {}".format(self.prefix,
                                                self.project_id,
                                                self.message)
        if self.backend_id is not None:
            return "{} backend '{}': {}".format(self.prefix,
                                                self.backend_id,
                                                self.message)
        return "{}: {}".format(self.prefix, self.message)


class ConfigurationException(AnnifException):
    """Exception raised when a project or backend is misconfigured."""

    prefix = "Misconfigured"


class NotSupportedException(AnnifException):
    """Exception raised when an operation is not supported by a project or
    backend."""

    prefix = "Not supported"
