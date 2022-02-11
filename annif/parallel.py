"""Parallel processing functionality for Annif"""


import multiprocessing
import multiprocessing.dummy


class BaseWorker:
    """Base class for workers that implement tasks executed via
    multiprocessing. The init method can be used to store data objects that
    are necessary for the operation. They will be stored in a class
    attribute that is accessible to the static worker method. The storage
    solution is inspired by this blog post:
    https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution # noqa
    """

    args = None

    @classmethod
    def init(cls, args):
        cls.args = args  # pragma: no cover


class ProjectSuggestMap:
    """A utility class that can be used to wrap one or more projects and
    provide a mapping method that converts Document objects to suggestions.
    Intended to be used with the multiprocessing module."""

    def __init__(
            self,
            registry,
            project_ids,
            backend_params,
            limit,
            threshold):
        self.registry = registry
        self.project_ids = project_ids
        self.backend_params = backend_params
        self.limit = limit
        self.threshold = threshold

    def suggest(self, doc):
        filtered_hits = {}
        for project_id in self.project_ids:
            project = self.registry.get_project(project_id)
            hits = project.suggest(doc.text, self.backend_params)
            filtered_hits[project_id] = hits.filter(
                project.subjects, self.limit, self.threshold)
        return (filtered_hits, doc.uris, doc.labels)


def get_pool(n_jobs):
    """return a suitable multiprocessing pool class, and the correct jobs
    argument for its constructor, for the given amount of parallel jobs"""

    if n_jobs < 1:
        n_jobs = None
        pool_class = multiprocessing.Pool
    elif n_jobs == 1:
        # use the dummy wrapper around threading to avoid subprocess overhead
        pool_class = multiprocessing.dummy.Pool
    else:
        pool_class = multiprocessing.Pool

    return n_jobs, pool_class
