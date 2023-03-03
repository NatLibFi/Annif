# Contributing to Annif

Contributions to Annif are very welcome!

This document aims to give you some helpful information when you wish to participate in
Annif development.

Typically you contribute by opening a new issue or offering modifications to the
codebase. Generally, in the case of non-trivial modifications, before submitting a
pull-request (PR) it is probably best to first discuss the topic in an issue.

When creating an issue, whether it is for a feature request/proposal, a bug report or a
question, you should first search the [existing issues (both open and
closed)](https://github.com/NatLibFi/Annif/issues?q=is%3Aissue) for your topic.
Feel free to comment existing issues to offer new details, ideas, opinions etc.

However, note that if you have a *general question about Annif or its usage in some
specific scenario or with a data-set*, please consider using the [annif-users mailing
list](https://groups.google.com/g/annif-users) (Google Groups) instead of opening an
issue.

## Creating an issue
If you don't find an existing issue for your topic, [open a new
one](https://github.com/NatLibFi/Annif/issues/new/choose). Please be clear in the title
and description, and provide all necessary information. In the case of a bug report the
provided information should aim to give a [minimal reproducible example of the
problem](https://stackoverflow.com/help/minimal-reproducible-example).

For readability, please format [code snippets as
code](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#quoting-code)
and also use other [markdown
formatting](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
where found appropriate.

## Contributing code
If you see an issue that you'd like to fix feel free to do so. If possible let us know
you're working on an issue by leaving a comment on it so we'll be able to avoid doing
the same work twice. This is especially useful if the issue has been marked for a
release (in a milestone with a version number) since it's more likely someone might be
already working on it.

### Installation for development
See [Development install in
README.md](https://github.com/NatLibFi/Annif/blob/master/README.md#development-install)
or use the [Docker image for
development](https://github.com/NatLibFi/Annif/wiki/Usage-with-Docker#using-docker-in-annif-development).

### Development flow
The development of Annif follows [GitHub
flow](https://guides.github.com/introduction/flow/). Feel free to [fork the
Annif-repository](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
for your changes. Some basic principles:

1. The `master` branch is always a working, deployable version of Annif. The code on the
   `master` branch will eventually be released as the next release.
2. All development happens on feature branches, whether branched from NatLibFi's origin
   or from a fork. Feature branches are normally named according to the issue they are
   addressing: e.g. `issue267-cli-analyze-to-suggest` which implements the change
   specified in issue [#267](https://github.com/NatLibFi/Annif/issues/267).
3. Feature branches are merged via *pull requests*. Opening a pull request signals the
   other developers that the feature is ready to be discussed and eventually merged.
   Pull requests should be marked with [draft
   status](https://github.blog/2019-02-14-introducing-draft-pull-requests/) if the
   developer knows that the code is not yet ready for merging but wants to start
   discussion. Also, various checks (tests in GitHub Actions, test coverage tools and
   static analyzer services) are run on pull requests and these may provide important
   feedback to the developer.
4. The pull request should have a clear description of the included changes, and if the
   PR is modified later, the description should be updated. Include a [linking
   keyword](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)
   targeting an issue when applicable, so when the PR is merged, the issue is
   automatically closed.
5. Feature branches should be deleted after the pull request has been merged.
6. A new release is made whenever some important changes have landed in `master`.
   Releases are intended to be frequent. See [Release
   process](https://github.com/NatLibFi/Annif/wiki/Release-process) for the details of
   making a release.

#### Commits
Try to produce a commit history that is easy to follow with meaningful commit messages.
See commit best practices e.g. in
[here](https://gist.github.com/luismts/495d982e8c5b1a0ced4a57cf3d93cf60#file-gitcommitbestpractices-md).

#### Branches

At any time, these branches typically exist:
* the `master` branch
* feature branches under development
* experimental branches that are not under active development but which we don't want to delete in case the code is later needed

#### Tags
Releases are tagged, e.g. `v0.40.0` and `v0.37.2`. The release tags are created using
the `bumpversion` tool, not manually. See [Release
process](https://github.com/NatLibFi/Annif/wiki/Release-process) for details.

### Unit tests
Generally, the aim is to cover every line of the codebase with the [unit
tests](https://github.com/NatLibFi/Annif/tree/master/tests). If you've added new
functionality or you've found out that the existing tests are lacking, we'd be happy if
you could provide additional tests to cover it. The development dependencies include
[`pytest`](https://docs.pytest.org/), which you can execute in the project root to run the unit tests:
```
pytest
```
To run only a subset of tests, you can pass a path to a tests file as an argument, e.g.: `pytest tests/test_analyzer.py`.
Also [`flake8`](https://flake8.pycqa.org/) checks are run together with the unit tests. It is best to verify that the unit tests
pass locally before pushing commits to GitHub repository.

When a (draft) PR is opened or new commits are pushed to a branch belonging to a PR, the
unit tests for the code are run in the [GitHub Actions CI/CD pipeline](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml). The tests are run
on all the minor versions of Python that Annif aims to support with varying
configurations of the optional dependencies, see the
[cicd.yaml](https://github.com/NatLibFi/Annif/blob/master/.github/workflows/cicd.yml)
for the pipeline setup.

### Code style

Annif code should follow the [Black
style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
and import statements should be [grouped and
ordered](https://peps.python.org/pep-0008/#imports). To achieve this, the Black and
[isort](https://pycqa.github.io/isort/) tools are included as development dependencies;
you can run `black .` and `isort .` in the project root to autoformat code. These tools
together with [`flake8`](https://flake8.pycqa.org/) are run also in GitHub Actions CI/CD pipeline checking the code
style compliance.

You can set up a [pre-commit
hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) to automate linting with
isort, Black and flake8 with every git commit by using the following in the file
`.git/hooks/pre-commit`, which should have execute permission set:
```bash
#!/bin/bash

set -e

isort . --check-only --diff
black . --check --diff
flake8
```
If the hook complains and intercepts the commit, you can run `isort .` and/or `black .` for an automatical fix.

Alternatively, you can set a pre-commit hook to also autoformat code using the
[pre-commit framework](https://pre-commit.com/), and configure it to use
[Black](https://black.readthedocs.io/en/stable/integrations/source_version_control.html)
and [isort](https://pycqa.github.io/isort/docs/configuration/pre-commit.html).

Other points:
- Names of the identifiers in the code (variables, functions, classes etc.) should be
meaningful. Do not use names of only single character.
 - Write docstrings for the entities you create. They end up in the [Annif's internal API
documentation](https://annif.readthedocs.io/en/latest/source/annif.html).

## Creating a new backend
Annif backend code is in the
[annif/backend](https://github.com/NatLibFi/Annif/tree/master/annif/backend) module.
Each backend is implemented as a subclass of `AnnifBackend`, or its more specific
subclass `AnnifLearningBackend` (for backends that support online learning) or
`BaseEnsembleBackend` (for backends that combine results from multiple projects).

A backend can define these key fields and methods:
* `name`: field for a name for the backend (a single word, all lowercase)
* `initialize` (optional): method setting up the necessary internal data structures
* `_train` (optional): method for training the model on a given document corpus
* `_suggest`: method for feeding a single document (text) and getting suggested subjects for it
* `_suggest_batch`: method for feeding a batch of documents (texts) and getting suggested subjects for each of them

It is only necessary to implement either `_suggest` or `_suggest_batch`, but
not both. Processing batches is often more efficient and should be used if
possible.

Learning backends additionally implement:

* `_learn`: method for continuing training the model on the given corpus
