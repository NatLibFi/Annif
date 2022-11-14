# Contributing

Contributions to Annif are very welcome!

This document aims to help you when you want to open a new issue or offer modifications
to the codebase with a pull-request (PR). Generally, in the case of non-trivial PRs, it
is probably best to first open a new issue so the topic can be discussed.

When creating an issue, whether it is for a feature request/proposal, a bug report or a
question, you should first search the [existing issues (both open and
closed)](https://github.com/NatLibFi/Annif/issues?q=is%3Aissue) for your topic.
Existing issues can be commented to offer new details or opinions.

Also note that if you have a general question about Annif or its usage in some specific
scenario or for a data-set, please consider using the [annif-users mailing
list](https://groups.google.com/g/annif-users) (Google Groups) instead of an issue.

## Creating an issue
If you don't find an existing issue for your topic, open a new one, and please be clear
in the title and description, and provide the necessary information. 
The provided information should aim to give a [minimal reproducible example of the
problem](https://stackoverflow.com/help/minimal-reproducible-example).

For readability, please format [code etc. blocks as code](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#quoting-code).

## Development flow
The development of Annif follows [GitHub flow](https://guides.github.com/introduction/flow/). Some basic principles:

1. The `master` branch is always a working, deployable version of Annif.  The code on the `master` branch will eventually be released as the next release.
2. All development happens on feature branches. Feature branches are normally named according to the issue they are addressing: e.g. `issue267-cli-analyze-to-suggest` which implements the change specified in issue [#267](https://github.com/NatLibFi/Annif/issues/267).
3. Feature branches are merged via *pull requests*. Opening a pull request signals the other developers that the feature is ready to be discussed and eventually merged. Pull requests should be marked with [draft status](https://github.blog/2019-02-14-introducing-draft-pull-requests/) if the developer knows that the code is not yet ready for merging but wants to start discussion. Also, various checks (tests in GitHub Actions, test coverage tools and static analyzer services) are run on pull requests and these may provide important feedback to the developer.
4. Feature branches should be deleted after the pull request has been merged.
5. A new release is made whenever some important changes have landed in `master`. Releases are intended to be frequent. See [Release process](https://github.com/NatLibFi/Annif/wiki/Release-process) for the details of making a release.

### Branches

At any time, these branches typically exist:
* the `master` branch
* feature branches under development
* experimental branches that are not under active development but which we don't want to delete in case the code is later needed

### Tags

Releases are tagged, e.g. `v0.40.0` and `v0.37.2`. The release tags are created using
the `bumpversion` tool, not manually. See §Release
process](https://github.com/NatLibFi/Annif/wiki/Release-process) for details.

## Code style

Annif code should follow the [Black
style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
and import statements should be [grouped and
ordered](https://peps.python.org/pep-0008/#imports). To achieve this, the Black and
[isort](https://pycqa.github.io/isort/) tools are included as development dependencies;
you can run `black .` and `isort .` in the project root to autoformat code. You can set
up a [pre-commit hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) to
automate linting with isort, Black and flake8 with every git commit by using the
following in the file `.git/hooks/pre-commit`, which should have execute permission set:
```bash
#!/bin/sh

isort . --check-only --diff
black . --check --diff
flake8
```
Alternatively, you can set a pre-commit hook to also autoformat code using the
[pre-commit framework](https://pre-commit.com/), and configure it to use
[Black](https://black.readthedocs.io/en/stable/integrations/source_version_control.html)
and [isort](https://pycqa.github.io/isort/docs/configuration/pre-commit.html).

## Creating a new backend
Annif backend code is in the [annif/backend](https://github.com/NatLibFi/Annif/tree/master/annif/backend) module. Each backend is implemented as a subclass of `AnnifBackend`, or its more specific subclass `AnnifLearningBackend` for backends that support online learning.

A backend must define these fields:

* `name`: a name for the backend (a single word, all lowercase)
* `needs_subject_index`: boolean value; defaults to False if not set; set to True if the backend makes use of the `SubjectIndex` passed as `project.subjects` to most methods
* `needs_subject_vectorizer`: boolean value; defaults to False if not set; set to True if the backend makes use of the `TfIdfVectorizer` passed as `project.vectorizer` to most methods

A backend needs to implement these key methods:

* `initialize` (optional): set up the necessary internal data structures
* `_train` (optional): train the model on a given document corpus
* `_suggest`: this is the key method that is given a single document (text) and returns suggested subjects

Learning backends additionally implement:

* `_learn`: continue training the model on the given corpus